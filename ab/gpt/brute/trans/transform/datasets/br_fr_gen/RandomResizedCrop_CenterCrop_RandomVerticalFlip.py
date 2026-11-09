import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.99), ratio=(1.07, 1.83)),
    transforms.CenterCrop(size=24),
    transforms.RandomVerticalFlip(p=0.43),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
