import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.12),
    transforms.RandomResizedCrop(size=32, scale=(0.54, 0.99), ratio=(1.29, 1.55)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
