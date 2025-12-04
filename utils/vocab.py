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
