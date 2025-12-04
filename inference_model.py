from tqdm import tqdm

import torch

from model import CRNN
from configs import Configs

@torch.no_grad()
def infer(model, dataloader, vocab, device):
    predictions, truth = [], []
    model.eval()

    for images, labels, _, label_lengths in tqdm(dataloader, desc="| Inferring"):
        images = images.to(device)

        outputs = model(images) # (W, B, C)

        predicted_indices = outputs.softmax(2).argmax(2)  # (W, B)
        predicted_indices = predicted_indices.permute(1, 0)  # (B, W)

        cumulative_sums = torch.cumsum(label_lengths, dim=0)
        start_index = 0

        for i, seq in enumerate(predicted_indices):
            end_index = cumulative_sums[i]
            predicted_text = vocab.decode(seq)
            true_text = vocab.decode(labels[start_index:end_index], collapse_repeats=False)

            predictions.append(predicted_text)
            truth.append(true_text)

            start_index = end_index
    
    return predictions, truth

def CER(predictions, truth):
    correct_chars = 0
    total_chars = 0

    for pred, label in zip(predictions, truth):
        correct_chars += sum(p == t for p, t in zip(pred, label))
        total_chars += len(label)
    
    return 1 - (correct_chars / total_chars)

def WER(predictions, truth):
    correct_words = sum(p == t for p, t in zip(predictions, truth))

    return 1 - (correct_words / len(truth))
