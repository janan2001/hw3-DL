import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List
import random
import spacy
import numpy as np
import time

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

# --- 2. Model Definition ---

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))
        # embedded = [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        input = input.unsqueeze(0)
        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <bos> token
        input = trg[0,:]

        for t in range(1, trg_len):

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs

# --- 3. Training Setup ---

INPUT_DIM = len(vocab_transform[SRC_LANGUAGE])
OUTPUT_DIM = len(vocab_transform[TGT_LANGUAGE])
ENC_EMB_DIM = 64 # Reduced for speed
DEC_EMB_DIM = 64 # Reduced for speed
HID_DIM = 128    # Reduced for speed
N_LAYERS = 1     # Reduced for speed
ENC_DROPOUT = 0.1 # Reduced
DEC_DROPOUT = 0.1 # Reduced

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

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
    for src_sample, trg_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        trg_batch.append(text_transform[TGT_LANGUAGE](trg_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX)
    return src_batch.to(device), trg_batch.to(device)

from torch.utils.data import DataLoader

BATCH_SIZE = 4 # Small batch size for mock data
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# --- 4. Training Loop ---

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    for i, (src, trg) in enumerate(iterator):
        optimizer.zero_grad()
        output = model(src, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            output = model(src, trg, 0) # turn off teacher forcing

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# --- 5. Inference / Chat Function ---

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    model.eval()

    if isinstance(sentence, str):
        if spacy_en:
            tokens = [token.text for token in spacy_en(sentence)]
        else:
            tokens = sentence.split()
    else:
        tokens = [token for token in sentence]

    # Handle unknown tokens
    unk_token_idx = 0 # src_field.get_default_index() returns 0 for UNK

    # Use the vocab directly (torchtext vocab is callable)
    token_indices = []
    for token in tokens:
        if token in src_field:
            token_indices.append(src_field[token])
        else:
            token_indices.append(unk_token_idx)

    tokens = [BOS_IDX] + token_indices + [EOS_IDX]

    src_tensor = torch.LongTensor(tokens).unsqueeze(1).to(device)

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    trg_indexes = [BOS_IDX]

    for i in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(trg_tensor, hidden, cell)

        pred_token = output.argmax(1).item()

        trg_indexes.append(pred_token)

        if pred_token == EOS_IDX:
            break

    trg_tokens = [trg_field.lookup_token(i) for i in trg_indexes]

    return trg_tokens[1:]

def chat(sentence):
    translation = translate_sentence(sentence, vocab_transform[SRC_LANGUAGE], vocab_transform[TGT_LANGUAGE], model, device)
    # Remove EOS token if present
    if translation[-1] == '<eos>':
        translation = translation[:-1]
    return " ".join(translation)

# --- Main Execution ---

if __name__ == "__main__":
    N_EPOCHS = 20
    CLIP = 1

    best_valid_loss = float('inf')

    print("Starting training...")
    start_time = time.time()
    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, val_dataloader, criterion)

        # We won't save the model file to avoid binary artifacts in the repo
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # torch.save(model.state_dict(), 'chatbot-model.pt')

        if (epoch+1) % 5 == 0:
            print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

    print(f"Training took {time.time() - start_time:.2f} seconds")

    print("\n--- Chatbot Demo ---")
    print("Bot: Hello! (Type 'quit' to exit)")
    test_sentences = ["Hello", "How are you?", "Good morning"]
    for sent in test_sentences:
        print(f"User: {sent}")
        response = chat(sent)
        print(f"Bot: {response}")

    print("\n(Note: The model is trained on a tiny mock dataset, so it will only respond correctly to specific phrases.)")
