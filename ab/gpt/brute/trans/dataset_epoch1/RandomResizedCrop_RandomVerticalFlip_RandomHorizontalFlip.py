import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.93), ratio=(0.79, 2.78)),
    transforms.RandomVerticalFlip(p=0.14),
    transforms.RandomHorizontalFlip(p=0.17),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
