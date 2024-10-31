
from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
import time
from io import open

import matplotlib

# if you are running on the gradx/ugradx/ another cluster,
# you will need the following line
# if you run on a local machine, you can comment it out
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim
from torch.autograd import Variable
import numpy as np
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid,
# it can be very easy to confict with other people's jobs.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15

def indexes_from_sentence(vocab, sentence):
    """
    Converts a sentence into a list of indices using the provided vocabulary.
    Each word is mapped to its index, and the EOS token is added at the end.
    """
    return [vocab.word2index.get(word, 0) for word in sentence.split()] + [EOS_index]


# import genism

class Vocab:
    """ This class handles the mapping between the words and their indicies
    """

    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# def get_pre_trained_embedding_mat(vocab, file, lan):
#   embeddings = []
#   model = gensim.models.KeyedVectors.load_word2vec_format(lan+"embeds")
#   for word in vocab.word2index:
#     embeddings.append(model.wv.get_vector(word, [0 for _ in range(100)]))
#   return embeddings


######################################################################


def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"),
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """ Creates the vocabs for each of the langues based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab


######################################################################

def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    # indexes = []
    tens = torch.ones(MAX_LENGTH, dtype=torch.long, device=device)
    for i, word in enumerate(sentence.split()):
        try:
            # indexes.append(vocab.word2index[word])
            tens[i] = vocab.word2index[word]
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    tens[len(sentence.split())] = EOS_index
    return tens.view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """Creates a tensor from a raw sentence pair and ensures batch compatibility."""
    input_tensor = tensor_from_sentence(src_vocab, pair[0]).unsqueeze(0)  # Add batch dimension
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1]).unsqueeze(0)  # Add batch dimension
    return input_tensor, target_tensor

######################################################################



def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence"""
    tens = torch.ones(len(sentence.split()), dtype=torch.long, device=device)
    for i, word in enumerate(sentence.split()):
        try:
            tens[i] = vocab.word2index[word]
        except KeyError:
            tens[i] = vocab.word2index["<UNK>"]  # Handle unknown words
    return tens

def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """Creates a tensor from a raw sentence pair."""
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor

def create_batch(pairs, src_vocab, tgt_vocab):
    """Creates batches of sentences."""
    input_tensors = []
    target_tensors = []
    for pair in pairs:
        input_tensor, target_tensor = tensors_from_pair(src_vocab, tgt_vocab, pair)
        input_tensors.append(input_tensor)
        target_tensors.append(target_tensor)

    # Pad the input and target tensors so that they all have the same length
    input_batch = pad_sequence(input_tensors, batch_first=True, padding_value=EOS_index)
    target_batch = pad_sequence(target_tensors, batch_first=True, padding_value=EOS_index)
    return input_batch, target_batch
def pad_seq(seq, max_length):
    """
    Pads a sequence with zeros up to the max_length.
    """
    return seq + [0] * (max_length - len(seq))



def prepare_batch_data(pairs, src_vocab, tgt_vocab, batch_size):
    batch_pairs = random.sample(pairs, batch_size)
    input_seqs = [indexes_from_sentence(src_vocab, pair[0]) for pair in batch_pairs]
    target_seqs = [indexes_from_sentence(tgt_vocab, pair[1]) for pair in batch_pairs]

    input_lengths = [len(seq) for seq in input_seqs]
    target_lengths = [len(seq) for seq in target_seqs]

    input_padded = [pad_seq(seq, max(input_lengths)) for seq in input_seqs]
    target_padded = [pad_seq(seq, max(target_lengths)) for seq in target_seqs]

    input_tensor = torch.LongTensor(input_padded).to(device)
    target_tensor = torch.LongTensor(target_padded).to(device)

    return input_tensor, input_lengths, target_tensor, target_lengths

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)  # Using batch_first for batches

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True, enforce_sorted=False)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)  # unpack back to padded
        return outputs, hidden

    def get_initial_hidden_state(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)



class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        # query: [batch_size, hidden_size]
        # keys: [batch_size, seq_len, hidden_size]
        scores = self.Va(torch.tanh(self.Wa(query).unsqueeze(1) + self.Ua(keys))).squeeze(-1)
        weights = F.softmax(scores, dim=1)  # normalize scores
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)  # context shape [batch_size, hidden_size]
        return context, weights


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        # input: [batch_size, 1]
        # hidden: [1, batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        embedded = self.dropout(self.embedding(input))
        context, attn_weights = self.attention(hidden[-1], encoder_outputs)  # Pass last hidden state to attention
        gru_input = torch.cat((embedded.squeeze(1), context), dim=1).unsqueeze(1)
        output, hidden = self.gru(gru_input, hidden)
        output = self.out(output.squeeze(1))
        return output, hidden, attn_weights

    def get_initial_hidden_state(self):
        return torch.zeros(1, self.hidden_size, device=device)


######################################################################





######################################################################



######################################################################

def translate(encoder, decoder, sentences, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """
    Translates a batch of sentences.
    """
    encoder.eval()
    decoder.eval()
    
    # Prepare input tensors
    input_tensors = [torch.tensor([src_vocab.word2index.get(word, 0) for word in sentence.split()] + [EOS_index], 
                                  dtype=torch.long) for sentence in sentences]
    input_lengths = [len(tensor) for tensor in input_tensors]
    input_padded = pad_sequence(input_tensors, batch_first=True, padding_value=EOS_index).to(device)

    # Encode
    with torch.no_grad():
        encoder_outputs, encoder_hidden = encoder(input_padded, input_lengths)
        batch_size = input_padded.size(0)
        decoder_input = torch.tensor([[SOS_index] for _ in range(batch_size)], device=device)  # [batch_size, 1]
        decoder_hidden = encoder_hidden

        # Store translation and attention results
        translated_words = [[] for _ in range(batch_size)]
        for _ in range(max_length):
            decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            decoder_input = topi.detach()  # Update input for the next time step

            for i in range(batch_size):
                if topi[i].item() == EOS_index:
                    translated_words[i].append(EOS_token)
                else:
                    translated_words[i].append(tgt_vocab.index2word.get(topi[i].item(), "<UNK>"))

    return translated_words

######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """
    Translate multiple sentence pairs from source to target in batch mode.
    """
    input_sentences = [pair[0] for pair in pairs]  # Source sentences from each pair
    
    # Use the `translate` function to get the output words
    output_words = translate(encoder, decoder, input_sentences, src_vocab, tgt_vocab, max_length)
    
    # Convert translated word lists into full sentences, removing EOS tokens
    output_sentences = [' '.join([word for word in words if word != EOS_token]) for words in output_words]

    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    """
    Translates `n` random sentences from the dataset and prints the input, target, and translation.
    """
    for _ in range(n):
        pair = random.choice(pairs)  # Randomly select a sentence pair
        print('Input:', pair[0])
        print('Target:', pair[1])
        
        # Translate the source sentence
        translated_words = translate(encoder, decoder, [pair[0]], src_vocab, tgt_vocab)
        translated_sentence = ' '.join([word for word in translated_words[0] if word != EOS_token])
        
        print('Predicted Translation:', translated_sentence)
        print('')


######################################################################

def show_attention(input_sentence, output_words, attentions, n):
    """visualize the attention mechanism. And save it to a file.
    Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
    You plots should include axis labels and a legend.
    you may want to use matplotlib.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig(f'{n}attention_plot.png')
    plt.show()
    # raise NotImplementedError


def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab, n):
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions, n)


def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').replace(SOS_token, '').strip().split())

def train_batch(input_tensor, target_tensor, input_lengths, target_lengths, encoder, decoder, optimizer, criterion):
    encoder.train()
    decoder.train()

    batch_size = input_tensor.size(0)
    encoder_hidden = encoder.get_initial_hidden_state(batch_size)
    encoder_outputs, encoder_hidden = encoder(input_tensor, input_lengths, encoder_hidden)

    decoder_input = torch.tensor([[SOS_index] * batch_size], device=device).transpose(0, 1)
    decoder_hidden = encoder_hidden
    all_decoder_outputs = torch.zeros(target_tensor.size(1), batch_size, decoder.output_size, device=device)

    for t in range(target_tensor.size(1)):
        decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
        all_decoder_outputs[t] = decoder_output
        decoder_input = target_tensor[:, t].unsqueeze(1)  # Teacher forcing

    # Calculate loss
    loss = criterion(all_decoder_outputs.view(-1, decoder.output_size), target_tensor.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


######################################################################
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int, help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=500000, type=int, help='total number of examples to train on')
    ap.add_argument('--batch_size', default=64, type=int, help='batch size for training')
    ap.add_argument('--print_every', default=100, type=int, help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=1000, type=int, help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.0001, type=float, help='initial learning rate')
    ap.add_argument('--src_lang', default='fr', help='Source (input) language code, e.g., "fr"')
    ap.add_argument('--tgt_lang', default='en', help='Target (output) language code, e.g., "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe', help='Training file with each line containing a source sentence followed by "|||", and a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe', help='Dev file with source-target sentence pairs')
    ap.add_argument('--test_file', default='data/fren.test.bpe', help='Test file with source sentences')
    ap.add_argument('--out_file', default='out2.txt', help='Output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1, help='Checkpoint file to start from')

    args = ap.parse_args()

    # Load vocab or initialize from scratch
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang, args.tgt_lang, args.train_file)

    # Initialize models
    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)

    # Load checkpoint weights if specified
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # Read data files
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    # Set up optimizer and loss function
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.AdamW(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # Load optimizer state if checkpointed
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    # Training loop
    while iter_num < args.n_iters:
        if iter_num % 100 == 0:
            print('iter_num:', iter_num)

        iter_num += 1
        
        # Sample a batch of training pairs
        batch_pairs = random.sample(train_pairs, args.batch_size)
        input_tensor, input_lengths, target_tensor, target_lengths = prepare_batch_data(batch_pairs, src_vocab, tgt_vocab, args.batch_size)
        
        # Train with batch
        loss = train_batch(input_tensor, target_tensor, input_lengths, target_lengths, encoder, decoder, optimizer, criterion)
        print_loss_total += loss

        # Checkpointing
        if iter_num % args.checkpoint_every == 0:
            state = {
                'iter_num': iter_num,
                'enc_state': encoder.state_dict(),
                'dec_state': decoder.state_dict(),
                'opt_state': optimizer.state_dict(),
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
            }
            filename = 'checkpoint_%010d.pt' % iter_num
            torch.save(state, filename)
            logging.debug('Checkpoint saved to %s', filename)

        # Logging
        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('Time since start: %s (iter: %d, %d%%) loss_avg: %.4f',
                         time.time() - start, iter_num, iter_num / args.n_iters * 100, print_loss_avg)
            
            # Translate a random batch from dev set for inspection
            translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)
            
            # Calculate BLEU score on dev set
            translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)
            references = [[clean(pair[1]).split()] for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)

    # Translate and save test set results
    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')

    # Visualize attention
    translate_and_show_attention("on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab, 1)
    translate_and_show_attention("j en suis contente .", encoder, decoder, src_vocab, tgt_vocab, 2)
    translate_and_show_attention("vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab, 3)
    translate_and_show_attention("c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab, 4)


if __name__ == '__main__':
    main()
