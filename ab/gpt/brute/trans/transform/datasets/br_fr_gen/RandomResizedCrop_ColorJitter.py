import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.76, 0.98), ratio=(1.3, 2.64)),
    transforms.ColorJitter(brightness=1.15, contrast=1.11, saturation=1.03, hue=0.02),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
