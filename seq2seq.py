# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
from timeit import default_timer as timer

# Constants
SRC_LANGUAGE = 'fr'
TGT_LANGUAGE = 'en'
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 32
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NUM_EPOCHS = 100

# Dataset class
class CustomSeq2SeqDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source_text = self.pairs[idx][0]
        target_text = self.pairs[idx][1]
        return (source_text, target_text)

# Load data
def load_data():
    tr_lines = open("./data/fren.train.bpe", encoding='utf-8').read().strip().split('\n')
    tr_pairs = [l.split('|||') for l in tr_lines]
    train_dataset = CustomSeq2SeqDataset(tr_pairs)

    d_lines = open("./data/fren.dev.bpe", encoding='utf-8').read().strip().split('\n')
    d_pairs = [l.split('|||') for l in d_lines]
    dev_dataset = CustomSeq2SeqDataset(d_pairs)

    return train_dataset, dev_dataset

# Tokenization and vocabulary
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        yield data_sample[language_index[language]].split()

def build_vocab(train_dataset):
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
    vocab_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_dataset, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)
        vocab_transform[ln].set_default_index(UNK_IDX)
    return vocab_transform

# Model components
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int, emb_size: int, nhead: int,
                 src_vocab_size: int, tgt_vocab_size: int, dim_feedforward: int = 512, dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src: torch.Tensor, trg: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor,
                src_padding_mask: torch.Tensor, tgt_padding_mask: torch.Tensor, memory_key_padding_mask: torch.Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, src_padding_mask, tgt_padding_mask,
                                memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)

# Helper functions
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

def beam_search_decode(model, src, src_mask, max_len, start_symbol, beam_width=3):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)

    # Initialize the beam with the start symbol
    beam = [(torch.tensor([start_symbol], device=DEVICE), 0.0)]

    for _ in range(max_len - 1):
        new_beam = []
        for seq, score in beam:
            if seq[-1].item() == EOS_IDX:
                new_beam.append((seq, score))
                continue

            tgt_mask = (generate_square_subsequent_mask(seq.size(0)).type(torch.bool)).to(DEVICE)
            out = model.decode(seq.unsqueeze(1), memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])

            # Get the top beam_width tokens and their log probabilities
            topk_prob, topk_idx = torch.topk(prob, beam_width)
            for i in range(beam_width):
                new_seq = torch.cat([seq, topk_idx[0][i].unsqueeze(0)])
                new_score = score + topk_prob[0][i].item()
                new_beam.append((new_seq, new_score))

        # Keep only the top beam_width sequences
        beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_width]

    # Return the sequence with the highest score
    return beam[0][0]

def translate(model: nn.Module, src_sentence: str, text_transform, vocab_transform, beam_width=3):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = beam_search_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX, beam_width=beam_width).flatten()
    # tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten() 
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


def translate(model: nn.Module, src_sentence: str, text_transform, vocab_transform):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")

# Training and evaluation
def train_epoch(model, optimizer, train_dataset, collate_fn):
    model.train()
    losses = 0
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        optimizer.zero_grad()
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()
        optimizer.step()
        losses += loss.item()
    return losses / len(train_dataloader)

def evaluate(model, dev_dataset, collate_fn):
    model.eval()
    losses = 0
    val_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        tgt_input = tgt[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(val_dataloader)

# Define the loss function
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Main function
def main():
    train_dataset, dev_dataset = load_data()
    vocab_transform = build_vocab(train_dataset)

    # Text transforms
    def tokenize(text):
        return text.split()

    def sequential_transforms(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    def tensor_transform(token_ids: List[int]):
        return torch.cat((torch.tensor([BOS_IDX]), torch.tensor(token_ids), torch.tensor([EOS_IDX])))

    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(tokenize, vocab_transform[ln], tensor_transform)

    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(text_transform[SRC_LANGUAGE](src_sample))
            tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample))
        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch

    # Model, loss, and optimizer
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    transformer = transformer.to(DEVICE)

    # Define loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # Training loop
    for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, train_dataset, collate_fn)
        end_time = timer()
        val_loss = evaluate(transformer, dev_dataset, collate_fn)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")
        if epoch % 10 == 0:
            refs = []
            targets = []
            for pr in dev_dataset:
                refs.append([pr[0].split()])
                targets.append(translate(transformer, pr[0], text_transform, vocab_transform).split())
            smoothing_function = SmoothingFunction().method1
            bleu_score_corpus = corpus_bleu(refs, targets, smoothing_function=smoothing_function)
            print("Corpus BLEU Score: ", bleu_score_corpus)

    # Translation example
    print(translate(transformer, "je n en suis pas trop convaincu .", text_transform, vocab_transform))

    # Translate test set
    test_lines = open("./data/fren.test.bpe", encoding='utf-8').readlines()
    with open("translations", "w") as fi:
        for line in test_lines:
            fi.write(translate(transformer, line.strip().split("|||")[0], text_transform, vocab_transform).strip() + "\n")

if __name__ == "__main__":
    main()