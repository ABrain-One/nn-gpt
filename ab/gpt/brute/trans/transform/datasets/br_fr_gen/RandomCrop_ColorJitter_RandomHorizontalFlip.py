import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=29),
    transforms.ColorJitter(brightness=0.98, contrast=0.84, saturation=1.01, hue=0.04),
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
