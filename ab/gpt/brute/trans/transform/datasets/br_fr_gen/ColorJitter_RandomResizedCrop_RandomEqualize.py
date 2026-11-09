import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.16, contrast=1.11, saturation=1.09, hue=0.02),
    transforms.RandomResizedCrop(size=32, scale=(0.59, 1.0), ratio=(0.96, 2.74)),
    transforms.RandomEqualize(p=0.42),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
