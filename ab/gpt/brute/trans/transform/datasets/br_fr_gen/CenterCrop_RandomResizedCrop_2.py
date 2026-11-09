import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=32),
    transforms.RandomResizedCrop(size=32, scale=(0.71, 0.92), ratio=(0.97, 1.96)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
