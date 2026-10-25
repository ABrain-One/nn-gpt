import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=24),
    transforms.CenterCrop(size=31),
    transforms.RandomResizedCrop(size=32, scale=(0.52, 0.92), ratio=(0.88, 1.95)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
