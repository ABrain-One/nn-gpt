import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.93), ratio=(1.02, 1.72)),
    transforms.RandomVerticalFlip(p=0.44),
    transforms.RandomInvert(p=0.59),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
