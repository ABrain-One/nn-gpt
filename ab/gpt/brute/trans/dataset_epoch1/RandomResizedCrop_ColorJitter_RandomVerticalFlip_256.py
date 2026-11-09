import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.68, 0.88), ratio=(1.11, 2.94)),
    transforms.ColorJitter(brightness=1.04, contrast=1.12, saturation=0.99, hue=0.01),
    transforms.RandomVerticalFlip(p=0.69),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
