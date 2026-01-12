import torch
import torch.nn as nn
import torch.nn.functional as F

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
