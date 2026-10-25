import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.95), ratio=(1.17, 1.53)),
    transforms.ColorJitter(brightness=0.86, contrast=1.14, saturation=0.81, hue=0.08),
    transforms.RandomRotation(degrees=11),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
