import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.17),
    transforms.RandomResizedCrop(size=32, scale=(0.55, 0.93), ratio=(0.92, 2.59)),
    transforms.RandomInvert(p=0.14),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
