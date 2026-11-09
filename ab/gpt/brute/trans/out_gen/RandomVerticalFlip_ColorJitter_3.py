import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.42),
    transforms.ColorJitter(brightness=1.12, contrast=1.01, saturation=1.03, hue=0.02),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
