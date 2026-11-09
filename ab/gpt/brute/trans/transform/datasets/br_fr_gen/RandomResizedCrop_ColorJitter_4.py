import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.81), ratio=(1.06, 2.08)),
    transforms.ColorJitter(brightness=1.16, contrast=0.94, saturation=1.12, hue=0.01),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
