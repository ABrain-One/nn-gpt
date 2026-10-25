import torch
import torchvision.transforms as transforms
from torchvision import functional as F

class SelectiveFlip:

    def __init__(self, p_vertical=0.5, p_horizontal=0.5):
        self.p_vertical = p_vertical
        self.p_horizontal = p_horizontal

    def __call__(self, img):
        if torch.rand(1) < self.p_vertical:
            return F.vflip(img)
        if torch.rand(1) < self.p_horizontal:
            return F.hflip(img)
        return img

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.6, 1.0), ratio=(0.8, 1.2)), transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), SelectiveFlip(), transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)), transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])])