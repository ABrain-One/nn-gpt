import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.82),
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.99), ratio=(1.27, 2.9)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
