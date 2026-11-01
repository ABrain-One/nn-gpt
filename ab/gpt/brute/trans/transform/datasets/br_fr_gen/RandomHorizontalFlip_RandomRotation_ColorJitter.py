import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.91, contrast=0.94, saturation=1.15, hue=0.06),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
