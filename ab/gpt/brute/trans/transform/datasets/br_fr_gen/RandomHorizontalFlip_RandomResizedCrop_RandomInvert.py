import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.67),
    transforms.RandomResizedCrop(size=32, scale=(0.54, 0.87), ratio=(1.07, 2.35)),
    transforms.RandomInvert(p=0.15),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
