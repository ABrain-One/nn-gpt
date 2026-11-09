import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=26),
    transforms.RandomResizedCrop(size=32, scale=(0.6, 0.97), ratio=(1.31, 1.37)),
    transforms.RandomInvert(p=0.61),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
