import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.13),
    transforms.ColorJitter(brightness=1.18, contrast=0.87, saturation=0.85, hue=0.08),
    transforms.RandomHorizontalFlip(p=0.22),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
