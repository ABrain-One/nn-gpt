import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.08, contrast=0.91, saturation=1.07, hue=0.1),
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.94), ratio=(0.99, 1.72)),
    transforms.RandomInvert(p=0.51),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
