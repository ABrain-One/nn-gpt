import torch
import torchvision.transforms as transforms
from torchvision.transforms import AutoAugmentPolicy


def transform(norm):
    return transforms.Compose([
    transforms.AutoAugment(policy=AutoAugmentPolicy.SVHN),     
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
