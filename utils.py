import torch
import sys
import tqdm
import numpy as np

def train_seq2seq(model, dl_train, optimizer, loss_fn, p_tf, GRAD_CLIP, BATCHES_PER_EPOCH):
    model.train()
    losses = []

    # We can't easily use tqdm with an iterator if we don't know the length,
    # but dl_train usually has a length if it's a DataLoader
    # The screenshot shows `for idx_batch, batch in enumerate(dl_train):`
    # but assumes BATCHES_PER_EPOCH limit? Or maybe it just iterates.
    # The screenshot snippet for train is not fully visible in the memory description,
    # but the eval one is. I'll follow standard practice and the variable names.

    # Assuming dl_train yields batches
    for idx_batch, batch in enumerate(dl_train):
        # Depending on how the batch is structured (torchtext vs custom collate)
        # The eval screenshot shows: x, x_len = batch.src; y, y_len = batch.trg
        # We need to match that structure.

        x, x_len = batch[0], batch[1] # Adapted for the tuple return from our custom collate
        y, y_len = batch[2], batch[3]

        optimizer.zero_grad()

        # Forward pass
        # model signature: forward(src, trg, p_tf, src_len)
        # Note: In the screenshot `y` is passed as target.
        # Usually y contains <sos> at start and <eos> at end.
        # The loss calculation usually ignores <sos> at index 0 of output vs target?
        # Standard: Output[t] predicts Target[t].
        # If Target is [<sos>, A, B, <eos>], Decoder Input starts with <sos>.
        # First output predicts A.
        # So we compare Output with Target[1:] (A, B, <eos>).
        # Let's see how `eval_seq2seq` handles it: `y_gt = y[1:, :] # drop <sos>`

        y_hat = model(x, y, p_tf, x_len)

        # y_hat: (max_len, batch, vocab)
        # y: (max_len, batch)

        # Loss calculation: flatten
        # We assume y includes <sos> at 0.
        # Output likely corresponds to prediction for next token.
        # If model.forward loop runs `max_len` times matching `trg` length:
        # Loop t=0: Input <sos>, Output predicts y[0]?? No, predicts y[0+1]?
        # Luong tutorial usually: Input <sos> -> Predict y[0] (which is first word).
        # Wait, if y is [<sos>, w1, w2, <eos>].
        # t=0: Input <sos>. Output predicts w1.
        # So y_hat[0] should be compared to y[1] (w1).
        # The screenshot `eval_seq2seq` does `y_gt = y[1:, :]`.
        # So we should do the same for training.

        # S, B, V = y_hat.shape
        # loss = loss_fn(y_hat.view(-1, V), y.view(-1))
        # ^ This would compare y_hat[0] (<sos>->w1) with y[0] (<sos>). WRONG.

        # Correct approach (standard):
        # y_hat matches length of y.
        # y_hat[t] is prediction for time t.
        # If we feed <sos> at start, y_hat[0] is prediction for first real word.
        # So we compare y_hat with y[1:].
        # But `model` implementation: `for t in range(max_len): ... decoder_input = trg[t]`.
        # If t=0, decoder_input = trg[0] (<sos>). Prediction is for trg[1].
        # So y_hat[0] is prediction for trg[1].
        # But the loop goes up to `max_len`.
        # If trg has length L. t goes 0..L-1.
        # Last input is trg[L-1] (<eos>). Prediction is for whatever comes after (padding?).
        # So y_hat has size L.
        # We compare y_hat[0...L-2] with trg[1...L-1].
        # Or y_hat[0...L-1] with trg[1...L] (if trg has valid next tokens).

        # Let's align with `eval_seq2seq`:
        # y_gt = y[1:, :]
        # So we ignore the first token of target (<sos>) for loss.
        # And we probably ignore the last output of y_hat if it corresponds to input <eos>.

        output_dim = y_hat.shape[-1]

        # Slicing
        # trg (y) includes <sos>.
        # y_hat predicts next token.
        # y_hat[0] predicts y[1].
        # ...
        # y_hat[len-2] predicts y[len-1] (<eos>).
        # y_hat[len-1] predicts post-<eos> (pad).

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
    # Note: Using `sys.stdout` for tqdm to avoid notebook issues if any
    # with tqdm.tqdm(total=len(dl_test), file=sys.stdout) as pbar:
    # Simplifying tqdm for script usage
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
        # Accuracy calculation per token? Or per sentence?
        # Screenshot says: `accuracies.append(torch.sum(y_gt == y_hat) / float(S))`
        # This implies per-batch token accuracy (summing all matches / S ... wait, S is sequence length?)
        # If it divides by S, it might be broadcasting or summing over B too?
        # `torch.sum(y_gt == y_hat)` returns scalar sum of matches.
        # `float(S)` is just one dimension.
        # It should probably be divided by total tokens (S * B).
        # But let's follow the screenshot logic if possible.
        # Screenshot: `accuracies.append(torch.sum(y_gt == y_hat) / float(S))`
        # If y_hat is (S, B), y_gt is (S, B).
        # This looks like sum of correct tokens / Sequence length.
        # This gives "correct tokens per sequence length" summed over batch?
        # Meaning: Average accuracy * Batch Size?
        # Or maybe S is total tokens? No, S, B, V = y_hat.shape. S is seq len.

        # Let's trust standard accuracy: Correct / Total
        correct = torch.sum(y_gt == y_hat_decoded).item()
        total = y_gt.numel()
        accuracies.append(correct / total)

        # pbar.update()

    return accuracies
