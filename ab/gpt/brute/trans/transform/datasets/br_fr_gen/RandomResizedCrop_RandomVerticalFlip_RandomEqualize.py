import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.71, 0.92), ratio=(0.8, 2.68)),
    transforms.RandomVerticalFlip(p=0.13),
    transforms.RandomEqualize(p=0.78),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
