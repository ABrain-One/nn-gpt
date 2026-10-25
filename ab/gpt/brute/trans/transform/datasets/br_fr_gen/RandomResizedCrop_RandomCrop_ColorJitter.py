import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.82), ratio=(1.2, 2.08)),
    transforms.RandomCrop(size=26),
    transforms.ColorJitter(brightness=0.88, contrast=1.01, saturation=0.81, hue=0.06),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
