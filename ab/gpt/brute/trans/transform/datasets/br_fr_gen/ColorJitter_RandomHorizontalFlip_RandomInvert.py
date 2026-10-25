import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.82, contrast=0.9, saturation=1.11, hue=0.01),
    transforms.RandomHorizontalFlip(p=0.43),
    transforms.RandomInvert(p=0.67),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
