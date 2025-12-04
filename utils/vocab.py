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
        
        if remove_blank:
            seq = [int(i) for i in seq if int(i) != 0]
        else:
            seq = [int(i) for i in seq]

        decoded = []
        prev_index = None

        for i in seq:
            if collapse_repeats:
                if i != prev_index:
                    decoded.append(self.idx2char[i])
            else:
                decoded.append(self.idx2char[i])
            prev_index = i
        
        return "".join(decoded)
