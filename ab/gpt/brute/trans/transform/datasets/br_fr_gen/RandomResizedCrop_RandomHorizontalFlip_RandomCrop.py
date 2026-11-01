import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.69, 0.86), ratio=(0.86, 2.46)),
    transforms.RandomHorizontalFlip(p=0.13),
    transforms.RandomCrop(size=24),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
