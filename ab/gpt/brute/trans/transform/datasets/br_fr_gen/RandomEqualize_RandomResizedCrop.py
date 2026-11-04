import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomEqualize(p=0.54),
    transforms.RandomResizedCrop(size=32, scale=(0.52, 0.92), ratio=(1.22, 2.13)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
