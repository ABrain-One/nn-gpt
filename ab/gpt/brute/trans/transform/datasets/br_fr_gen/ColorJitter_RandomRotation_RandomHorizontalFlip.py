import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.87, contrast=0.86, saturation=1.03, hue=0.1),
    transforms.RandomRotation(degrees=0),
    transforms.RandomHorizontalFlip(p=0.43),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
