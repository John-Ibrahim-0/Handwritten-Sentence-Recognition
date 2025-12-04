from tqdm import tqdm

import torch

@torch.no_grad()
def infer(model, dataloader, vocab, device):
    all_predictions, all_truth = [], []
    model.eval()

    for images, labels, _, label_lengths in tqdm(dataloader, desc="| Inferring"):
        images = images.to(device)

        outputs = model(images) # (W, B, C)

        batch_predictions, batch_truth = vocab.decode_batch(outputs, labels, label_lengths)

        all_predictions.extend(batch_predictions)
        all_truth.extend(batch_truth)
    
    return all_predictions, all_truth

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
