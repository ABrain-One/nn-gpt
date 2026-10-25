import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.72),
    transforms.RandomAutocontrast(p=0.35),
    transforms.ColorJitter(brightness=0.96, contrast=1.06, saturation=0.99, hue=0.02),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
