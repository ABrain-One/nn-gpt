import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.02, contrast=1.08, saturation=1.05, hue=0.06),
    transforms.RandomCrop(size=24),
    transforms.RandomResizedCrop(size=32, scale=(0.59, 0.87), ratio=(1.0, 2.5)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
