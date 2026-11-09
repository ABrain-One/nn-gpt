import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAutocontrast(p=0.36),
    transforms.ColorJitter(brightness=0.94, contrast=1.08, saturation=1.19, hue=0.07),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
