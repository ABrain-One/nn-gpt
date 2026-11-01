import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=30),
    transforms.RandomInvert(p=0.33),
    transforms.RandomResizedCrop(size=32, scale=(0.61, 0.97), ratio=(0.82, 2.79)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
