import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=32),
    transforms.RandomHorizontalFlip(p=0.78),
    transforms.RandomResizedCrop(size=32, scale=(0.59, 1.0), ratio=(1.28, 2.6)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
