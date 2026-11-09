import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.RandomResizedCrop(size=32, scale=(0.68, 0.84), ratio=(0.77, 2.31)),
    transforms.RandomCrop(size=28),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
