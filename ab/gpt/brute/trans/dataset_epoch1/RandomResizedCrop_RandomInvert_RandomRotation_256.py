import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.56, 0.86), ratio=(0.88, 2.58)),
    transforms.RandomInvert(p=0.24),
    transforms.RandomRotation(degrees=10),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
