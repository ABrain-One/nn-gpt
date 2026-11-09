import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.98), ratio=(1.3, 2.64)),
    transforms.ColorJitter(brightness=1.0, contrast=1.09, saturation=0.9, hue=0.08),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
