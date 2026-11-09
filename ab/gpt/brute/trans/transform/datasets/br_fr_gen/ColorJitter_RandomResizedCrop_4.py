import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.17, contrast=1.07, saturation=1.15, hue=0.03),
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.95), ratio=(1.31, 2.76)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
