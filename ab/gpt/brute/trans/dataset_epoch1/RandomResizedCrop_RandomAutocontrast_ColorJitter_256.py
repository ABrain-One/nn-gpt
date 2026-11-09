import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.83), ratio=(0.97, 2.2)),
    transforms.RandomAutocontrast(p=0.85),
    transforms.ColorJitter(brightness=0.84, contrast=0.95, saturation=1.06, hue=0.03),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
