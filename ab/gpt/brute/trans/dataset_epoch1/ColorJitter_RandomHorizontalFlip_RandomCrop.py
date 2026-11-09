import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.15, contrast=1.17, saturation=0.83, hue=0.06),
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomCrop(size=26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
