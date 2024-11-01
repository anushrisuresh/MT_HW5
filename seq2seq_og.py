#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
Students *MAY NOT* view the above tutorial or use it as a reference in any way.
"""

from __future__ import unicode_literals, print_function, division

import time
import random
import logging
import argparse
import matplotlib
from io import open
from tqdm import tqdm

# if you are running on the gradx/ugradx/ another cluster,
# you will need the following line
# if you run on a local machine, you can comment it out
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15


class Vocab:
    """ This class handles the mapping between the words and their indices
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
    """ Creates the vocabs for each of the languages based on the training corpus.
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
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


def sentence_indices(lang, sentence):
    """
    function to convert sentence from string to int based on the vocab indices
    """
    return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_index]


######################################################################

def sequence_padding(seq, max_len, pad_const=1):
    if len(seq) >= max_len:
        return seq[:max_len]

    padding = [pad_const] * (max_len - len(seq))
    return seq + padding

def mini_batching(batch_size, sentence_pairs, i_vocab, t_vocab):
    """
    function to create mini batches out of the data using random shuffle
    """
    inp_seq = []
    target_seq = []

    sentence_pairs = list(random.sample(sentence_pairs, batch_size))
    for input, output in sentence_pairs:
        inp_seq.append(sentence_indices(i_vocab, input))
        target_seq.append(sentence_indices(t_vocab, output))

    # Sort the sequences by their length
    seq_pairs = sorted(zip(inp_seq, target_seq), key=lambda x: len(x[0]), reverse=True)
    inp_seq, target_seq = tuple(zip(*seq_pairs))

    # Pad the input and target sequences
    pad_input = []
    for input_seq in inp_seq:
        pad_input.append(sequence_padding(input_seq, MAX_LENGTH, EOS_index))
    pad_target = []
    for target_seq in target_seq:
        pad_target.append(sequence_padding(target_seq, MAX_LENGTH, EOS_index))

    # Convert the padded sequences to tensors
    inp_seq = torch.Tensor(pad_input)
    target_seq = torch.Tensor(pad_target)

    return inp_seq, target_seq


######################################################################


class EncoderRNN(nn.Module):
    """the class for the encoder RNN
    """

    def __init__(self, input_size, hidden_size, batch_size):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # Initializing the layers
        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=EOS_index)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)  

    def forward(self, input_batch, hidden, test=False):
        """runs the forward pass of the encoder
        returns the output and the hidden state
        """
        if test:
            cur_b_size = 1
        else:
            cur_b_size = self.batch_size

        embedded = self.embedding(input_batch.long().to(device))
        # Reshape the embedded tensor to [batch_size, 1, hidden_size]
        embedded = embedded.view(cur_b_size, 1, -1)
        lstm_output, hidden = self.lstm(embedded, hidden)
        outputs = lstm_output

        return outputs, hidden


    def get_initial_hidden_state(self):
        return None  


class AttnDecoderRNN(nn.Module):
    """the class for the decoder
    """

    def __init__(self, hidden_size, output_size, D, dropout_p=0.2, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.D = D

        self.dropout = nn.Dropout(self.dropout_p)

        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, dropout=dropout_p, batch_first=True)  
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.attention = nn.Linear(self.hidden_size * 2, 2 * self.D + 1)
        self.combine_attn = nn.Linear(self.hidden_size * 2, self.hidden_size)


    def forward(self, input, hidden, encoder_outputs, window):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state, and attn_weights

        Dropout (self.dropout) should be applied to the word embeddings.
        """

        embedded = self.dropout(self.embedding(input))  

        attn_input = torch.cat((embedded[:, 0, :], hidden[0][0, :, :]), dim=1)
        attn_logits = self.attention(attn_input)
        attn_weights = self.softmax(attn_logits)
        applied_attn = torch.bmm(attn_weights[:, window[0]:window[1]].unsqueeze(1), encoder_outputs)
        output = torch.cat((embedded, applied_attn), 2)
        output = self.relu(self.combine_attn(output))

        output, hidden = self.lstm(output, hidden)
        output = F.log_softmax(self.out(output[:, 0, :]), dim=1)

        return output, hidden, attn_weights


    def get_initial_hidden_state(self):
        return None  


######################################################################

def train(input_batch, target_batch, encoder, decoder, optimizer, criterion, w_size, max_length=MAX_LENGTH):
    """
    function to train the nn
    """
    # make sure the encoder and decoder are in training mode so dropout is applied
    encoder.train()
    decoder.train()

    # Initialize variables for training loop
    encoder_hidden = encoder.get_initial_hidden_state()
    optimizer.zero_grad()
    loss = 0
    start_time = time.time()

    # Determine input and target lengths
    input_len = input_batch[0].shape[0]
    t_length = target_batch[0].shape[0]
    batch_size = input_batch.shape[0]

    # Initialize encoder outputs tensor
    encoder_outputs = torch.zeros((batch_size, max_length, encoder.hidden_size), device=device)

    # Iterate over the input sequence, processing each word at a time
    for ei in range(input_len):
        # Pass the current input word to the encoder
        encoder_out, encoder_hidden = encoder(input_batch[:, ei].view(-1, 1), encoder_hidden)
        encoder_outputs[:, ei, :] = encoder_out.squeeze(dim=1)

    # Initialize the decoder input with the SOS index
    decoder_inp = torch.tensor(SOS_index, device=device).view(1, 1).expand(batch_size, 1)

    # Initialize the decoder hidden state with the final encoder hidden state
    decoder_hidden = encoder_hidden

    for i in range(t_length):
        # Calculate the window boundaries and context window lengths
        window_l = max(i - w_size, 0)
        window_r = min(t_length, i + w_size + 1)
        if window_l == 0:
            l = 0 - (i - w_size)
        else:
            l = 0

        if (i + w_size + 1) > t_length:
            r = t_length - (i + w_size + 1)
        else:
            r = 2 * w_size + 1

        # Extract relevant encoder outputs based on the window boundaries
        relevant_encoder_outputs = encoder_outputs[:, window_l:window_r, :]
        decoder_out, decoder_hidden, decoder_attention = decoder(decoder_inp, decoder_hidden,
                                                                    relevant_encoder_outputs, (l, r))

        # Obtain the top prediction from the decoder output
        topv, topi = decoder_out.topk(1, dim=1)
        decoder_inp = topi.squeeze().detach().reshape((-1, 1))

        # Calculate the loss for the current step and add it to the accumulated loss
        loss += criterion(decoder_out, target_batch[:, i].long().to(device))


    loss.backward()
    optimizer.step()

    target_lengths = [i.size(0) for i in target_batch]
    total_length = sum(target_lengths)
    batch_size = len(target_batch)
    t_length = total_length / batch_size


    return loss.item()


######################################################################


def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, w_size, max_length=MAX_LENGTH):
    """
    runs translation, returns the output and attention
    """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    batch_size = 1  

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        input_len = input_tensor.size()[0]
        encoder_hidden = encoder.get_initial_hidden_state()

        encoder_outputs = torch.zeros(batch_size, max_length, encoder.hidden_size, device=device)

        for ei in range(input_len):
            encoder_out, encoder_hidden = encoder(input_tensor[ei].reshape((-1, 1)), encoder_hidden, test=True)
            encoder_outputs[:, ei, :] = encoder_out.squeeze(dim=1)
        decoder_inp = torch.tensor([[SOS_index] * batch_size], device=device).reshape(-1, 1)
        decoder_hidden = encoder_hidden  

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, 2 * w_size + 1)

        for di in range(max_length):
            window_l = max(di - w_size, 0)
            l = 0
            if window_l == 0:
                l = 0 - (di - w_size)
            else:
                l = 0
            window_r = min(max_length, di + w_size + 1)
            r = 0
            if di + w_size + 1 > max_length:
                r = max_length - (di + w_size + 1)
            else:
                r = 2 * w_size + 1
            decoder_out, decoder_hidden, decoder_attention = decoder(decoder_inp, decoder_hidden,
                                                                        encoder_outputs[:, window_l:window_r, :],
                                                                        (l, r))
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_out.topk(1, dim=1)
            decoder_inp = topi.squeeze().detach().reshape((-1, 1))

            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tgt_vocab.index2word[topi.item()])

        return decoded_words, decoder_attentions[:di + 1]


######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, w_size, max_num_sentences=None,
                        max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab, w_size)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#
def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, w_size, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab, w_size)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

def show_attention(input_sentence, output_words, attentions, fig, ax):
    """visualize the attention mechanism. And save it to a file.
    Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
    You plots should include axis labels and a legend.
    you may want to use matplotlib.
    """

    ax.matshow(attentions.numpy(), cmap='bone')

    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    plt.show()

def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab, w_size, fig, ax):
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab, w_size)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions, fig, ax)


def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=100000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=100, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=1000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=int,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--batch_size', default=256, type=int,
                    help='batch size for the mini batches')
    ap.add_argument('--w_size', default=3, type=int,
                    help='window size for local attn')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')

    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')
    ap.add_argument('--fig', default='output.png',
                    help='output figure for attention')

    args = ap.parse_args()

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size, args.batch_size).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, D=args.w_size, dropout_p=0.1).to(device)

    # encoder/decoder weights are randomly initialized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    for _ in tqdm(range(args.n_iters)):
        iter_num += 1

        input_batch, target_batch = mini_batching(args.batch_size, train_pairs, src_vocab, tgt_vocab)

        loss = train(input_batch, target_batch, encoder, decoder, optimizer, criterion, args.w_size)

        print_loss_total += loss

        if iter_num % args.checkpoint_every == 0:
            state = {'iter_num': iter_num,
                     'enc_state': encoder.state_dict(),
                     'dec_state': decoder.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     'src_vocab': src_vocab,
                     'tgt_vocab': tgt_vocab,
                     }
            filename = 'state_%010d.pt' % iter_num
            torch.save(state, filename)
            logging.debug('wrote checkpoint to %s', filename)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                         time.time() - start,
                         iter_num,
                         iter_num / args.n_iters * 100,
                         print_loss_avg)
            # translate from the dev set
            translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, args.w_size, n=2)
            translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab,
                                                       args.w_size)

            references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)

    # translate test set and write to file
    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab, args.w_size)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')

    # Visualizing Attention
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    translate_and_show_attention("on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab, args.w_size, fig, ax1)
    translate_and_show_attention("j en suis contente .", encoder, decoder, src_vocab, tgt_vocab, args.w_size, fig, ax2)
    translate_and_show_attention("vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab, args.w_size, fig, ax3)
    translate_and_show_attention("c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab, args.w_size, fig, ax4)

    plt.savefig(args.fig)


if __name__ == '__main__':
    main()
