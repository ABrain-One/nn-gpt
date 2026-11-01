import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.85),
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.83), ratio=(0.99, 1.77)),
    transforms.RandomHorizontalFlip(p=0.14),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
