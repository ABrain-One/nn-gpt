import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.77),
    transforms.ColorJitter(brightness=0.97, contrast=1.18, saturation=1.12, hue=0.04),
    transforms.RandomResizedCrop(size=32, scale=(0.69, 0.98), ratio=(0.84, 2.94)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
