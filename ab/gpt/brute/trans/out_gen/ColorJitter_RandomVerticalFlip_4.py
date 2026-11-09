import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.16, contrast=1.12, saturation=1.0, hue=0.06),
    transforms.RandomVerticalFlip(p=0.19),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
