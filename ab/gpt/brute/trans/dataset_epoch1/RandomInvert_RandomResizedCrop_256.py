import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.14),
    transforms.RandomResizedCrop(size=32, scale=(0.52, 1.0), ratio=(0.85, 2.92)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
