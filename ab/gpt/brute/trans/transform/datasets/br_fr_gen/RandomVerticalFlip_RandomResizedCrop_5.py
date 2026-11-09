import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.63),
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.93), ratio=(0.78, 2.98)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
