import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=26),
    transforms.RandomAutocontrast(p=0.16),
    transforms.ColorJitter(brightness=0.97, contrast=1.08, saturation=1.12, hue=0.05),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
