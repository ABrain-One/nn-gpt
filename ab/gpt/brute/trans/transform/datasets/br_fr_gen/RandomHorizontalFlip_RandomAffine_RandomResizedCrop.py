import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.11),
    transforms.RandomAffine(degrees=21, translate=(0.01, 0.16), scale=(0.98, 1.47), shear=(0.05, 8.52)),
    transforms.RandomResizedCrop(size=32, scale=(0.62, 0.84), ratio=(1.1, 1.42)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
