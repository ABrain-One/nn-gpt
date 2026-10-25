import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.36),
    transforms.RandomResizedCrop(size=32, scale=(0.72, 0.96), ratio=(0.9, 2.96)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
