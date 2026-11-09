import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=27),
    transforms.RandomResizedCrop(size=32, scale=(0.71, 0.91), ratio=(1.31, 2.2)),
    transforms.RandomInvert(p=0.64),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
