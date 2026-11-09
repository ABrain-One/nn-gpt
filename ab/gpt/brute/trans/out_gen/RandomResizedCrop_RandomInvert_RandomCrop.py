import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.51, 0.94), ratio=(1.25, 2.62)),
    transforms.RandomInvert(p=0.82),
    transforms.RandomCrop(size=26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
