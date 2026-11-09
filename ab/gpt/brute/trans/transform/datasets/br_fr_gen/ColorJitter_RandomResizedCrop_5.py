import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.19, contrast=0.93, saturation=1.0, hue=0.06),
    transforms.RandomResizedCrop(size=32, scale=(0.51, 0.91), ratio=(1.17, 1.77)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
