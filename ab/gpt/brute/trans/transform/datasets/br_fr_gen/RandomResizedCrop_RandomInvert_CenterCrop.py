import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.71, 0.97), ratio=(1.32, 1.43)),
    transforms.RandomInvert(p=0.19),
    transforms.CenterCrop(size=30),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
