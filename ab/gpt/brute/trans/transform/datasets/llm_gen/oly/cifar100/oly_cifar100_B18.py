import torch
import torchvision.transforms as transforms
from torchvision import functional as F

def apply_autocontrast_p(p=0.89):

    def _apply_autocontrast(img):
        if torch.rand(1) < p:
            img = F.autocontrast(img)
        return img
    return _apply_autocontrast

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(231, 103, 189), padding_mode='constant'), transforms.RandomResizedCrop(64, scale=(0.5, 1.0), ratio=(0.8, 1.2)), transforms.ColorJitter(0.85, 1.12, 1.0, 0.0), apply_autocontrast_p(), transforms.RandomApply([transforms.Equalize()], p=0.89), transforms.ToTensor(), transforms.Normalize(*norm)])