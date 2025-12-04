from PIL import Image

import torch
import torch.nn.functional as F
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

def collate_fn(batch, model):
    images, labels = zip(*batch)

    # get max width in the batch
    max_width = max([image.shape[2] for image in images]) # image shape: (C, H, W)

    padded_images = []

    for image in images:
        _, _, w = image.shape
        pad_width = max_width - w

        padded = F.pad(image, (0, pad_width, 0, 0), value=0.0)
        padded_images.append(padded)
    
    # stack images into tensor (B, C, H, W)
    images_tensor = torch.stack(padded_images)

    # convert labels to tensors
    label_tensors = [torch.tensor(label, dtype=torch.long) for label in labels]
    labels_tensor = torch.cat(label_tensors)

    # length of each label
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

    with torch.no_grad():
        dummy_image = images_tensor[0].unsqueeze(0)  # (1, C, H, W)
        cnn_out = model.cnn(dummy_image) # pass through CNN only
        seq_len = cnn_out.size(-1)  # width dimension after CNN
    
    image_lengths = torch.tensor([seq_len] * len(images), dtype=torch.long)

    return images_tensor, labels_tensor, image_lengths, label_lengths
