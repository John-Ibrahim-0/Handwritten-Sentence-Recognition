from PIL import Image
from torch.utils.data import Dataset

class HandwrittenSentenceDataset(Dataset):
    def __init__(self, image_paths, labels, transforms=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("L")

        if self.transforms:
            image = self.transforms(image)
        
        label = self.labels[index]

        return image, label
