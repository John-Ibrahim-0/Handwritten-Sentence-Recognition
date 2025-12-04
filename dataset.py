from PIL import Image
from torch.utils.data import Dataset

from configs import Configs

class HandwrittenSentenceDataset(Dataset):
    def __init__(self, image_paths, labels, vocab, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.vocab = vocab
        self.transforms = transforms

        configs = Configs()
        BLANK = configs.BLANK_LABEL
        
        self.char2idx = {BLANK: 0}
        for i, char in enumerate(vocab, start=1):
            self.char2idx[char] = i
        
        self.idx2char = {i: char for char, i in self.char2idx.items()}
    
    def encode_label(self, label):
        return [self.char2idx[char] for char in label if char in self.char2idx]

    def decode_sequence(self, seq):
        return "".join([self.idx2char[i] for i in seq if i in self.idx2char])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("L")

        if self.transforms:
            image = self.transforms(image)
        
        label = self.encode_label(self.labels[index])

        return image, label
