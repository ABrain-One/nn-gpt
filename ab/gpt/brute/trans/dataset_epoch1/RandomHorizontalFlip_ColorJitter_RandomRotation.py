import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.86),
    transforms.ColorJitter(brightness=1.0, contrast=1.2, saturation=1.03, hue=0.09),
    transforms.RandomRotation(degrees=15),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
