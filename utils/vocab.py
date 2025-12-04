import torch

class Vocab:
    def __init__(self, vocab, blank="-"):
        self.blank = blank
        
        self.char2idx = {blank: 0}
        self.char2idx.update({c: i+1 for i, c in enumerate(vocab)})

        self.idx2char = {i: c for c, i in self.char2idx.items()}
    
    def __len__(self):
        return len(self.char2idx)
    
    def encode(self, label):
        return [self.char2idx[c] for c in label if c in self.char2idx]
    
    def decode(self, seq, collapse_repeats=True, remove_blank=True):
        if isinstance(seq, torch.Tensor):
            seq = seq.detach().cpu().tolist()
        
        collapsed = []
        prev_index = None

        for i in seq:
            if collapse_repeats:
                if i != prev_index:
                    collapsed.append(i)
            else:
                collapsed.append(i)
            prev_index = i

        if remove_blank:
            collapsed = [i for i in collapsed if i != 0]

        return "".join(self.idx2char[i] for i in collapsed)

    def decode_batch(self, logits, labels, label_lengths):
        predicted_indices = logits.softmax(2).argmax(2)  # (W, B)
        predicted_indices = predicted_indices.permute(1, 0)  # (B, W)

        predictions, truth = [], []

        cumulative_sums = torch.cumsum(label_lengths, dim=0)
        start_index = 0

        for i, seq in enumerate(predicted_indices):
            end_index = cumulative_sums[i]
            predicted_text = self.decode(seq)
            true_text = self.decode(labels[start_index:end_index], collapse_repeats=False)

            predictions.append(predicted_text)
            truth.append(true_text)

            start_index = end_index
        
        return predictions, truth
