import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import HandwrittenSentenceDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

LABEL_PATH = os.path.join("data", "ascii", "sentences.txt")
IMAGE_FOLDER_PATH = os.path.join("data", "sentences")

images, labels = [], []

with open(LABEL_PATH, "r") as file:
    for line in tqdm(file.readlines(), desc="Loading dataset"):
        # skip documentation lines
        if line.startswith("#"):
            continue

        line = line.split()

        # skip sentences with segmentation errors
        if line[2] == "err":
            continue

        image_folder_1 = line[0][:3]
        image_folder_2 = line[0][3:8].rstrip("-")
        image_file_name = line[0] + ".png"

        image_path = os.path.join(IMAGE_FOLDER_PATH, image_folder_1, image_folder_2, image_file_name)
        label = line[-1].rstrip("\n").replace("|", " ")

        images.append(image_path)
        labels.append(label)

print(len(images), "images found.")
print(len(labels), "labels found.")
