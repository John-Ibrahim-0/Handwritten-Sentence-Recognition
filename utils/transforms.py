from torchvision.transforms import functional as F

class ResizeHeight:
    def __init__(self, height):
        self.height = height

    def __call__(self, image):
        w, h = image.size
        new_w = int(w * (self.height / h))
        return F.resize(image, (self.height, new_w))
