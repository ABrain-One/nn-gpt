import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 1.0), ratio=(1.32, 2.23)),
    transforms.CenterCrop(size=25),
    transforms.ColorJitter(brightness=1.08, contrast=1.15, saturation=1.15, hue=0.07),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
