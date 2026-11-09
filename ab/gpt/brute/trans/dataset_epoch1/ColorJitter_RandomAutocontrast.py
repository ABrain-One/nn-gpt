import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.18, contrast=0.86, saturation=0.99, hue=0.02),
    transforms.RandomAutocontrast(p=0.17),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
