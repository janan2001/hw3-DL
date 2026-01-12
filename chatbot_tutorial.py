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

# Import new models and utils
from models import EncoderRNN, Attn, LuongAttnDecoderRNN, Seq2Seq
from utils import train_seq2seq, eval_seq2seq

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
# Note: models.py assumed SOS=2. Let's align.
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
    # Although EncoderRNN handles sorting internally usually via wrapper or we do it here.
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

# --- 2. Model Definition (Replacing with screenshot models) ---

INPUT_DIM = len(vocab_transform[SRC_LANGUAGE])
OUTPUT_DIM = len(vocab_transform[TGT_LANGUAGE])
# Screenshot settings might be 64/64/128 etc.
EMB_DIM = 64
HID_DIM = 128
NUM_LAYERS = 2 # Usually 2 in tutorial
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
