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
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)

        if encoder_hidden.size(0) == self.decoder.n_layers * 2:
            decoder_hidden0 = encoder_hidden.view(self.decoder.n_layers, 2, 1, -1).sum(dim=1)
        else:
            decoder_hidden0 = encoder_hidden[:self.decoder.n_layers]

        # Initialize beams
        # (tokens, hidden, total_logp, token_logps, finished)
        beams_sen = [([], decoder_hidden0, 0.0, [], False)]
        completed_sen = []

        for _ in range(max_length):
            new_beams = []
            if all(finished for (_, _, _, _, finished) in beams_sen):
                break

            # Expand each beam
            for tokens, hidden, total_logp, token_logps, finished in beams_sen:
                if finished:
                    # already ended with EOS -> keep it without expanding
                    new_beams.append((tokens, hidden, total_logp, token_logps, True))
                    continue

                # Decoder input = last generated token, or SOS if none yet
                if len(tokens) == 0:
                    decoder_input = torch.tensor([[self.SOS_token]], device=self.device, dtype=torch.long)
                else:
                    decoder_input = torch.tensor([[tokens[-1]]], device=self.device, dtype=torch.long)

                # One decoder step
                decoder_output, next_hidden = self.decoder(decoder_input, hidden, encoder_outputs)

                # Convert logits -> log-probs (Using F.log_softmax because model outputs logits)
                log_probs = F.log_softmax(decoder_output, dim=1)  # (1, vocab_size), stable

                # Take top-k next tokens for this beam
                # Note: if we have B beams, expanding each by beam_size results in B*K candidates.
                # The user's snippet expands each beam by K and then prunes to K total.
                top_logp, top_idx = torch.topk(log_probs, self.beam_size, dim=1)

                # Create new hypotheses from these k expansions
                for j in range(self.beam_size):
                    next_token = top_idx[0, j].item()      # predicted token id (int)
                    next_logp = top_logp[0, j].item()      # log P(token | history, x)

                    new_tokens = tokens + [next_token]     # append token to the sequence
                    new_token_logps = token_logps + [next_logp]
                    new_total = total_logp + next_logp     # accumulate log-prob sum
                    new_finished = (next_token == self.EOS_token)

                    hyp = (new_tokens, next_hidden, new_total, new_token_logps, new_finished)

                    # If finished, store separately; otherwise keep for next step
                    if new_finished:
                        completed_sen.append(hyp)
                    else:
                        new_beams.append(hyp)

            # Keep only the best k active beams (highest total log-prob)
            if len(new_beams) > 0:
                new_beams.sort(key=lambda x: x[2], reverse=True)
                beams_sen = new_beams[:self.beam_size]
            else:
                # If everything finished this step, keep current beams as-is
                break

        # ---------------------------
        # 5) Choose best final hypothesis
        # Prefer completed if any; else best unfinished
        # ---------------------------
        if len(completed_sen) > 0:
            completed_sen.sort(key=lambda x: x[2], reverse=True)
            best_tokens, _, _, best_token_logps, _ = completed_sen[0]
        else:
            beams_sen.sort(key=lambda x: x[2], reverse=True)
            best_tokens, _, _, best_token_logps, _ = beams_sen[0]

        # Convert Python lists to tensors (like GreedySearchDecoder returns)
        best_tokens_tensor = torch.tensor(best_tokens, device=self.device, dtype=torch.long)
        best_scores_tensor = torch.tensor(best_token_logps, device=self.device, dtype=torch.float)

        return best_tokens_tensor, best_scores_tensor
