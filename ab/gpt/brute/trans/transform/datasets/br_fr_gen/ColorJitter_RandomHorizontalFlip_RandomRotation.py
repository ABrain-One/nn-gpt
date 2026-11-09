import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.98, contrast=1.17, saturation=1.06, hue=0.02),
    transforms.RandomHorizontalFlip(p=0.16),
    transforms.RandomRotation(degrees=6),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
