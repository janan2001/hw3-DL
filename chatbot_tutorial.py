import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
import random
import spacy
import numpy as np
import time
import sys

# Set seed for reproducibility
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Check for device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 1. Data Preparation ---

# Mock Data (Conversational: Message -> Response)
mock_data = [
    ("Hello", "Hi there!"),
    ("How are you?", "I am doing well, thank you."),
    ("What is your name?", "I am a chatbot tutorial bot."),
    ("What are you doing?", "I am learning to chat."),
    ("Goodbye", "See you later!"),
    ("Hi", "Hello!"),
    ("Good morning", "Good morning to you too."),
    ("Good evening", "Good evening!"),
    ("Thanks", "You are welcome."),
    ("Who created you?", "I was created by a developer."),
    ("Can you help me?", "Yes, I can help you."),
    ("What time is it?", "I do not know the time."),
    ("Where are you from?", "I live in the cloud."),
    ("Do you like pizza?", "I do not eat, but pizza sounds good."),
    ("Are you real?", "I am a real program."),
]

# We repeat the mock data to simulate a larger dataset - REDUCED for faster testing
train_data = mock_data * 20
val_data = mock_data * 2
test_data = mock_data * 2

# Load Spacy models once
try:
    # Use English for both source and target since it's a chatbot in English
    spacy_en = spacy.load('en_core_web_sm')
    tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
except OSError:
    print("Spacy models not found, using basic split tokenizer")
    spacy_en = None
    tokenizer_en = lambda x: x.split()

def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    # Both languages use the same tokenizer for this chatbot
    for data_sample in data_iter:
        if language == SRC_LANGUAGE:
            yield tokenizer_en(data_sample[0])
        else:
            yield tokenizer_en(data_sample[1])

SRC_LANGUAGE = 'src'
TGT_LANGUAGE = 'trg'

# Special tokens
# standard: unk=0, pad=1, bos=2, eos=3
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Build Vocab
vocab_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Create a generator for yield_tokens to avoid consuming the list
    vocab_transform[ln] = build_vocab_from_iterator(
        yield_tokens(train_data, ln),
        min_freq=1,
        specials=special_symbols,
        special_first=True
    )

# Set default index for unknown tokens
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

# Helper function for collation
from torch.nn.utils.rnn import pad_sequence

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(tokenizer_en,
                                               vocab_transform[ln],
                                               tensor_transform)

def collate_fn(batch):
    src_batch, trg_batch = [], []
    src_lens, trg_lens = [], []

    for src_sample, trg_sample in batch:
        src_tokens = text_transform[SRC_LANGUAGE](src_sample.rstrip("\n"))
        trg_tokens = text_transform[TGT_LANGUAGE](trg_sample.rstrip("\n"))

        src_batch.append(src_tokens)
        trg_batch.append(trg_tokens)

        src_lens.append(len(src_tokens))
        trg_lens.append(len(trg_tokens))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX)

    # Sort batch by source length (descending) for pack_padded_sequence
    # Standard PyTorch pack_padded_sequence needs sorted input if enforce_sorted=True (default).

    src_lens = torch.LongTensor(src_lens)
    trg_lens = torch.LongTensor(trg_lens)

    sorted_lens, sorted_indices = torch.sort(src_lens, descending=True)
    src_batch_sorted = src_batch[:, sorted_indices]
    trg_batch_sorted = trg_batch[:, sorted_indices]
    trg_lens_sorted = trg_lens[sorted_indices]

    return src_batch_sorted.to(device), sorted_lens.to(device), trg_batch_sorted.to(device), trg_lens_sorted.to(device)

from torch.utils.data import DataLoader

BATCH_SIZE = 16 # Small batch size for mock data
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False, drop_last=True)

# --- 2. Model Definition ---

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        # because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # input_seq: (max_length, batch_size)
        # input_lengths: (batch_size)
        # hidden: (n_layers * num_directions, batch_size, hidden_size)

        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)

        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu())

        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)

        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]

        # Return output and final hidden state
        # outputs: (max_length, batch_size, hidden_size)
        # hidden: (n_layers * num_directions, batch_size, hidden_size)
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # hidden: (1, batch_size, hidden_size)
        # encoder_outputs: (max_length, batch_size, hidden_size)

        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # input_step: (1, batch_size)
        # last_hidden: (n_layers, batch_size, hidden_size)
        # encoder_outputs: (max_length, batch_size, hidden_size)

        # Note: we run this one step (word) at a time

        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)

        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)

        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        # output = F.softmax(output, dim=1) # Removed for CrossEntropyLoss which expects logits

        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, p_tf, src_len):
        # src: (max_length, batch_size)
        # trg: (max_length, batch_size) - target sentence
        # p_tf: teacher forcing ratio
        # src_len: (batch_size) - lengths of source sentences

        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size

        # Tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, vocab_size).to(src.device)

        # Encoder forward pass
        encoder_outputs, encoder_hidden = self.encoder(src, src_len)

        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Initialize decoder input with SOS_token
        # Assuming trg[0] is SOS. If so, we use it.
        # But we can also force SOS_token if we know it.
        decoder_input = trg[0].unsqueeze(0)

        # Iterate over target sequence
        # We start from t=0.
        # But `outputs[t]` stores the prediction.
        # If t=0, decoder_input is SOS. output[0] predicts the first real word.
        # The loop range should cover the full length.

        for t in range(max_len):
            decoder_output, decoder_hidden = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            outputs[t] = decoder_output

            # Teacher forcing: use actual target as next input
            use_teacher_forcing = True if torch.rand(1).item() < p_tf else False

            if use_teacher_forcing:
                # Next input is the target word at `t+1`
                # If t is the last index, we can't get t+1.
                if t + 1 < max_len:
                    decoder_input = trg[t + 1].unsqueeze(0)
                else:
                    # End of sequence, doesn't matter much but keep consistent
                    pass
            else:
                # Next input is decoder's own current prediction
                _, topi = decoder_output.topk(1)
                decoder_input = topi.transpose(0, 1).detach() # (1, batch_size)

        return outputs

# --- 3. Training & Evaluation Utils ---

def train_seq2seq(model, dl_train, optimizer, loss_fn, p_tf, GRAD_CLIP, BATCHES_PER_EPOCH):
    model.train()
    losses = []

    # Assuming dl_train yields batches
    for idx_batch, batch in enumerate(dl_train):

        x, x_len = batch[0], batch[1] # Adapted for the tuple return from our custom collate
        y, y_len = batch[2], batch[3]

        optimizer.zero_grad()

        # Forward pass
        # model signature: forward(src, trg, p_tf, src_len)
        y_hat = model(x, y, p_tf, x_len)

        output_dim = y_hat.shape[-1]

        # Slicing
        # trg (y) includes <sos>.
        # y_hat matches length of y.
        # y_hat[t] is prediction for time t.
        # If we feed <sos> at start, y_hat[0] is prediction for first real word.
        # So we compare y_hat with y[1:].

        loss = loss_fn(y_hat[:-1].view(-1, output_dim), y[1:].view(-1))

        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        losses.append(loss.item())

        if idx_batch >= BATCHES_PER_EPOCH:
            break

    return losses

def eval_seq2seq(model, dl_test):
    accuracies = []

    for idx_batch, batch in enumerate(dl_test):
        x, x_len = batch[0], batch[1]
        y, y_len = batch[2], batch[3]

        with torch.no_grad():
            # Note: no teacher forcing in eval
            y_hat = model(x, y, p_tf=0, src_len=x_len)

        S, B, V = y_hat.shape
        y_gt = y[1:, :] # drop <sos>
        y_hat_decoded = torch.argmax(y_hat, dim=2) # greedy-sample (S, B, V) -> (S, B)

        # We need to slice y_hat_decoded to match y_gt length
        y_hat_decoded = y_hat_decoded[:-1]

        # Compare prediction to ground truth
        correct = torch.sum(y_gt == y_hat_decoded).item()
        total = y_gt.numel()
        accuracies.append(correct / total)

    return accuracies

# --- 4. Model Instantiation ---

INPUT_DIM = len(vocab_transform[SRC_LANGUAGE])
OUTPUT_DIM = len(vocab_transform[TGT_LANGUAGE])
EMB_DIM = 64
HID_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.1

print(f"Building models with V_src={INPUT_DIM}, V_tgt={OUTPUT_DIM}...")

# Initialize embeddings
enc_embedding = nn.Embedding(INPUT_DIM, HID_DIM)
dec_embedding = nn.Embedding(OUTPUT_DIM, HID_DIM)

enc = EncoderRNN(HID_DIM, enc_embedding, NUM_LAYERS, DROPOUT)
dec = LuongAttnDecoderRNN('dot', dec_embedding, HID_DIM, OUTPUT_DIM, NUM_LAYERS, DROPOUT)

seq2seq_model = Seq2Seq(enc, dec).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)

seq2seq_model.apply(init_weights)

optimizer = optim.Adam(seq2seq_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

# --- 5. Inference / Chat Function ---

def chat(sentence, max_len=10):
    seq2seq_model.eval()

    # Preprocess
    tokens = text_transform[SRC_LANGUAGE](sentence)
    src_len = torch.LongTensor([len(tokens)]).to(device)
    src_tensor = tokens.unsqueeze(1).to(device) # (len, 1)

    with torch.no_grad():
        encoder_outputs, encoder_hidden = seq2seq_model.encoder(src_tensor, src_len)
        decoder_hidden = encoder_hidden[:seq2seq_model.decoder.n_layers]

        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * BOS_IDX

        decoded_words = []

        for i in range(max_len):
            decoder_output, decoder_hidden = seq2seq_model.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            _, top_index = decoder_output.topk(1)
            # top_index: (1, 1)

            if top_index.item() == EOS_IDX:
                break

            decoded_words.append(vocab_transform[TGT_LANGUAGE].lookup_token(top_index.item()))
            decoder_input = top_index.transpose(0, 1)

    return " ".join(decoded_words)

# --- Main Execution ---

if __name__ == "__main__":
    EPOCHS = 20
    GRAD_CLIP = 5.0 # Typical value
    BATCHES_PER_EPOCH = len(train_dataloader)

    print("Starting training...")
    start_time = time.time()

    losses = []
    accuracies = []

    for idx_epoch in range(EPOCHS):
        # Linearly decay amount of teacher forcing for the first 20 epochs
        # p_tf = 1 - min((idx_epoch / 20), 1)
        # Or fixed for stability on mock data
        p_tf = 1.0 if idx_epoch < 5 else 0.5

        epoch_losses = train_seq2seq(seq2seq_model, train_dataloader, optimizer, criterion, p_tf, GRAD_CLIP, BATCHES_PER_EPOCH)
        losses.extend(epoch_losses)

        # Validation
        epoch_accs = eval_seq2seq(seq2seq_model, val_dataloader)
        accuracies.extend(epoch_accs)

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        avg_acc = np.mean(epoch_accs) if epoch_accs else 0

        if (idx_epoch+1) % 1 == 0:
            print(f'Epoch: {idx_epoch+1:02} | Train Loss: {avg_loss:.3f} | Val Acc: {avg_acc:.3f} | p_tf: {p_tf}')

    print(f"Training took {time.time() - start_time:.2f} seconds")

    print("\n--- Chatbot Demo ---")
    print("Bot: Hello! (Type 'quit' to exit)")
    test_sentences = ["Hello", "How are you?", "Good morning"]
    for sent in test_sentences:
        print(f"User: {sent}")
        response = chat(sent)
        print(f"Bot: {response}")

    print("\n(Note: The model is trained on a tiny mock dataset, so it will only respond correctly to specific phrases.)")
