#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Students *MAY NOT* view the above tutorial or use it as a reference in any way.
"""

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

from tqdm import tqdm

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid,
# it can be very easy to confict with other people's jobs.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# from __future__ import unicode_literals, print_function, division
import argparse
import logging
import random
import time
from io import open

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim
import numpy as np

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"
SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15
BATCH_SIZE = 32


class Vocab:
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2

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


def split_lines(input_file):
    logging.info("Reading lines of %s...", input_file)
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)
    train_pairs = split_lines(train_file)
    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])
    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)
    return src_vocab, tgt_vocab


def tensor_from_sentence(vocab, sentence):
    indexes = [vocab.word2index.get(word, EOS_index) for word in sentence.split()]
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size=BATCH_SIZE):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.lstm(embedded, hidden)
        return output, hidden

    def init_hidden(self, batch_size=BATCH_SIZE):
        return (torch.zeros(1, batch_size, self.hidden_size, device=device),
                torch.zeros(1, batch_size, self.hidden_size, device=device))


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.transpose(1, 2), keys)
        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.attention = BahdanauAttention(hidden_size)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True)  # batch_first=True for batching
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        # Embed input tokens
        embedded = self.dropout(self.embedding(input))  # Shape: (batch_size, 1, hidden_size)
        
        # Calculate attention context
        context, attn_weights = self.attention(hidden[0].transpose(0, 1), encoder_outputs)
        
        # Ensure dimensions match for concatenation
        if embedded.dim() == 2:
            embedded = embedded.unsqueeze(1)  # (batch_size, 1, hidden_size)
        if context.dim() == 2:
            context = context.unsqueeze(1)    # (batch_size, 1, hidden_size)
        
        # Concatenate `embedded` and `context` along the last dimension
        lstm_input = torch.cat((embedded, context), dim=2)
        
        # Forward pass through LSTM
        output, hidden = self.lstm(lstm_input, hidden)
        
        # Output layer and log softmax
        output = F.log_softmax(self.out(output.squeeze(1)), dim=1)
        return output, hidden, attn_weights


def train(input_tensors, target_tensors, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_hidden = encoder.init_hidden(batch_size=BATCH_SIZE)
    input_tensors = torch.nn.utils.rnn.pad_sequence(input_tensors, batch_first=True, padding_value=EOS_index)
    target_tensors = torch.nn.utils.rnn.pad_sequence(target_tensors, batch_first=True, padding_value=EOS_index)

    encoder_outputs, encoder_hidden = encoder(input_tensors, encoder_hidden)

    decoder_input = torch.tensor([SOS_index] * BATCH_SIZE, device=device).unsqueeze(1)  # Batch-first
    decoder_hidden = encoder_hidden

    loss = 0
    target_length = target_tensors.size(1)

    for di in range(target_length):
        decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
        
        # Compute loss only on non-pad tokens
        loss += criterion(decoder_output, target_tensors[:, di])
        
        decoder_input = target_tensors[:, di].unsqueeze(1)  # Set next input to current target token

    loss = loss / target_length  # Normalize by length
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss.item()


# Adjust the main function to create batches and update training logic.
# Full `main` function update is similar to the original one but uses the modified functions.




######################################################################
def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence).unsqueeze(0)  # add batch dimension
        input_length = input_tensor.size(1)
        
        # Initialize encoder hidden state
        encoder_hidden = encoder.init_hidden(batch_size=1)
        encoder_outputs, encoder_hidden = encoder(input_tensor, encoder_hidden)

        # Initialize the decoder input with SOS and set the initial hidden state
        decoder_input = torch.tensor([[SOS_index]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        actual_max_length = encoder_outputs.size(1)
        decoder_attentions = torch.zeros(actual_max_length, actual_max_length)

        for di in range(actual_max_length):
            decoder_output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di, :attn_weights.size(1)] = attn_weights.data.squeeze()
            
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tgt_vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    """
    Translate a list of sentences and return the translated output sentences.
    """
    encoder.eval()
    decoder.eval()

    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, _ = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words).replace(SOS_token, '').replace(EOS_token, '').strip()
        output_sentences.append(output_sentence)
    
    return output_sentences


def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    """
    Translate n random sentences from the provided pairs and print the results.
    """
    for _ in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, _ = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words).replace(SOS_token, '').replace(EOS_token, '').strip()
        print('<', output_sentence)
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


######################################################################


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int, help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=5000, type=int, help='total number of examples to train on')
    ap.add_argument('--print_every', default=100, type=int, help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=1000, type=int, help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=float, help='initial learning rate')
    ap.add_argument('--src_lang', default='fr', help='Source (input) language code, e.g., "fr"')
    ap.add_argument('--tgt_lang', default='en', help='Target (output) language code, e.g., "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe', help='training file with source and target sentences separated by "|||"')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe', help='dev file with source and target sentences')
    ap.add_argument('--test_file', default='data/fren.test.bpe', help='test file with source and target sentences')
    ap.add_argument('--out_file', default='out2.txt', help='output file for test translations')
    args = ap.parse_args()

    # Build vocabulary from training data
    src_vocab, tgt_vocab = make_vocabs(args.src_lang, args.tgt_lang, args.train_file)
    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)

    # Load training, dev, and test data
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    # Initialize optimizer and loss function
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.initial_learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.initial_learning_rate)
    criterion = nn.NLLLoss(ignore_index=EOS_index)

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    for iter_num in range(1, args.n_iters + 1):
        # Prepare a random batch
        batch_pairs = [random.choice(train_pairs) for _ in range(BATCH_SIZE)]
        input_tensors = [tensor_from_sentence(src_vocab, pair[0]) for pair in batch_pairs]
        target_tensors = [tensor_from_sentence(tgt_vocab, pair[1]) for pair in batch_pairs]

        # Train on batch
        loss = train(input_tensors, target_tensors, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('Time since start: %s (iter: %d, %.2f%%) loss_avg: %.4f',
                         time.time() - start, iter_num, iter_num / args.n_iters * 100, print_loss_avg)
            # Translate some dev sentences
            translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)

        if iter_num % args.checkpoint_every == 0:
            # Save checkpoint
            state = {
                'iter_num': iter_num,
                'encoder_state': encoder.state_dict(),
                'decoder_state': decoder.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab
            }
            checkpoint_path = f'checkpoint_{iter_num}.pt'
            torch.save(state, checkpoint_path)
            logging.debug('Checkpoint saved at %s', checkpoint_path)

    # Translate and write the test set to file
    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab)
    with open(args.out_file, 'w', encoding='utf-8') as f:
        for sentence in translated_sentences:
            f.write(sentence + '\n')

    # Visualize attention for a few example sentences
    sample_sentences = ["on p@@ eu@@ t me faire confiance .", "j en suis contente .", "vous etes tres genti@@ ls ."]
    for idx, sent in enumerate(sample_sentences, start=1):
        translate_and_show_attention(sent, encoder, decoder, src_vocab, tgt_vocab, idx)

if __name__ == '__main__':
    main()




if __name__ == '__main__':
    main()
