import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomEqualize(p=0.57),
    transforms.RandomResizedCrop(size=32, scale=(0.72, 0.95), ratio=(1.2, 2.17)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
