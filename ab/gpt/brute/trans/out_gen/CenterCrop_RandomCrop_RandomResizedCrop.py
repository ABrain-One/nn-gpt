import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=24),
    transforms.RandomCrop(size=24),
    transforms.RandomResizedCrop(size=32, scale=(0.76, 0.93), ratio=(0.81, 2.96)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
