import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=32),
    transforms.RandomInvert(p=0.66),
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.97), ratio=(0.88, 1.88)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
