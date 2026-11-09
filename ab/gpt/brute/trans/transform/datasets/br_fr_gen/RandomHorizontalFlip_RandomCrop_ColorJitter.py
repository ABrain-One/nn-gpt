import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.58),
    transforms.RandomCrop(size=32),
    transforms.ColorJitter(brightness=0.83, contrast=0.83, saturation=1.11, hue=0.03),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
