import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.85, contrast=0.94, saturation=1.12, hue=0.05),
    transforms.RandomHorizontalFlip(p=0.16),
    transforms.RandomResizedCrop(size=32, scale=(0.69, 0.86), ratio=(1.17, 2.51)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
