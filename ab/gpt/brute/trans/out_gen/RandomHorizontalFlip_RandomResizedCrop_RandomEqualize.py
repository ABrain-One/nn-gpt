import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.8), ratio=(0.86, 1.96)),
    transforms.RandomEqualize(p=0.82),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
