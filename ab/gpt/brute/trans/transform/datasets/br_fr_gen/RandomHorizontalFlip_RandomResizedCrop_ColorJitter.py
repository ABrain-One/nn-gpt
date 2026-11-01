import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.38),
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.87), ratio=(1.09, 2.36)),
    transforms.ColorJitter(brightness=0.98, contrast=0.98, saturation=1.06, hue=0.02),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
