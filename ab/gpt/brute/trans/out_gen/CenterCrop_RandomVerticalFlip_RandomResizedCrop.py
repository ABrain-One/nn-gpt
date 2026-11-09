import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=26),
    transforms.RandomVerticalFlip(p=0.79),
    transforms.RandomResizedCrop(size=32, scale=(0.75, 0.84), ratio=(0.77, 2.45)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
