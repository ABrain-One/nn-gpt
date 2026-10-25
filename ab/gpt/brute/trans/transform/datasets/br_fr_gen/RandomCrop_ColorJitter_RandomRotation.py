import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.ColorJitter(brightness=0.83, contrast=0.9, saturation=1.15, hue=0.07),
    transforms.RandomRotation(degrees=25),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
