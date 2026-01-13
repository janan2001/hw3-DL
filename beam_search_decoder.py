import torch
import torch.nn as nn
import torch.nn.functional as F

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, device, SOS_token):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.SOS_token = SOS_token

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=self.device, dtype=torch.long) * self.SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            # Note: decoder_output is (batch_size, vocab_size), here batch=1
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

class BeamSearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, device, SOS_token, EOS_token, beam_size=5):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.SOS_token = SOS_token
        self.EOS_token = EOS_token
        self.beam_size = beam_size

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)

        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        # encoder_hidden: (n_layers * num_directions, batch_size, hidden_size)
        # decoder_hidden: (n_layers, batch_size, hidden_size)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Initialize the beam with the start sequence
        # Each hypothesis is a tuple: (last_token, hidden_state, sequence, score)
        # But for efficiency, we can keep track of sequences and scores separately

        # We start with one hypothesis: [SOS], score=0
        # For the loop, we track:
        # - current_tokens: list of last tokens for each hypothesis (size k)
        # - current_hidden: hidden states for each hypothesis (size k, n_layers, 1, hidden)
        # - sequences: list of token lists
        # - scores: list of cumulative log-probs

        # Initial input
        decoder_input = torch.ones(1, 1, device=self.device, dtype=torch.long) * self.SOS_token

        # Initial set of beams
        # sequence, score, hidden state
        # sequence is a list of tensors
        beams = [([], 0.0, decoder_hidden)]

        for _ in range(max_length):
            new_beams = []

            for seq, score, hidden in beams:
                if len(seq) > 0 and seq[-1].item() == self.EOS_token:
                    # If this beam already ended, keep it (don't expand)
                    new_beams.append((seq, score, hidden))
                    continue

                # Prepare input for this beam
                if len(seq) == 0:
                    inp = torch.tensor([[self.SOS_token]], device=self.device, dtype=torch.long)
                else:
                    inp = seq[-1].view(1, 1)

                # Forward pass
                # inp: (1, 1)
                # hidden: (n_layers, 1, hidden_size)
                # encoder_outputs: (seq_len, 1, hidden_size) - need to expand this if we batched, but we loop here

                decoder_output, next_hidden = self.decoder(inp, hidden, encoder_outputs)
                # decoder_output: (1, vocab_size) - logits (since we removed Softmax in models.py)

                # Apply Log Softmax to get log-probabilities
                log_probs = F.log_softmax(decoder_output, dim=1)

                # Get top k candidates
                # If we are at the very beginning (just SOS), we pick top k from this one.
                # If we have B beams, we might pick top k from each?
                # Standard Beam Search:
                # Expand all B current beams -> B * Vocab possibilities.
                # Calculate new scores.
                # Prune to top k best among ALL B * Vocab possibilities.

                # Optimization: We usually just take top k from each to limit size, then prune global list.
                top_scores, top_indices = torch.topk(log_probs, self.beam_size)

                for i in range(self.beam_size):
                    token = top_indices[0][i]
                    token_score = top_scores[0][i].item()

                    new_seq = seq + [token]
                    new_score = score + token_score
                    new_beams.append((new_seq, new_score, next_hidden))

            # Sort all new beams by score (descending)
            new_beams.sort(key=lambda x: x[1], reverse=True)

            # Keep top k
            beams = new_beams[:self.beam_size]

            # Check if all top k are finished (optional optimization)
            all_finished = True
            for seq, _, _ in beams:
                if len(seq) == 0 or seq[-1].item() != self.EOS_token:
                    all_finished = False
                    break
            if all_finished:
                break

        # Return best sequence
        best_seq, best_score, _ = beams[0]

        # Convert list of tensors to single tensor
        if len(best_seq) > 0:
            all_tokens = torch.stack(best_seq)
            # Scores: we only tracked cumulative. The prompt asks for "collections of word tokens and scores".
            # Greedy decoder returns list of scores per token.
            # Beam search usually maximizes total score.
            # We can just return the total score or reconstruct per-token scores if needed.
            # Given the Greedy signature `all_scores` (tensor), let's just return a tensor of the total score or similar.
            # Or simplified: just return the tokens.
            # The prompt says: "Return the best decoded token sequence (and optionally its scores)."

            # Let's mock per-token scores as 0 for now or just return the cumulative scalar as a 1-element tensor?
            # Greedy returns `all_scores` as vector of probs.
            # We'll just return the best tokens.
            return all_tokens, torch.tensor([best_score], device=self.device)
        else:
            return torch.tensor([], device=self.device), torch.tensor([0.0], device=self.device)
