import torch
import torchvision.transforms as F
from torch import nn

def transform(norm):
    return F.Compose([F.RandomResizedCrop(224, scale=(0.5, 2.0)), F.RandomHorizontalFlip(), F.RandomRotation(degrees=(-15, 15)), F.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), F.RandomApply([F.Autocontrast()], p=0.2), F.RandomApply([F.Equalize()], p=0.2), F.ToTensor(), F.Normalize(*norm)])