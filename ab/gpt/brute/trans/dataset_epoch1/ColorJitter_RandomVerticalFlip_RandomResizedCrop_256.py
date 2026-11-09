import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.92, contrast=0.98, saturation=1.06, hue=0.01),
    transforms.RandomVerticalFlip(p=0.59),
    transforms.RandomResizedCrop(size=32, scale=(0.79, 0.92), ratio=(1.08, 1.93)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
