import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.0, contrast=0.88, saturation=1.03, hue=0.03),
    transforms.RandomVerticalFlip(p=0.61),
    transforms.RandomAutocontrast(p=0.11),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
