import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.83),
    transforms.RandomResizedCrop(size=32, scale=(0.71, 0.82), ratio=(0.77, 1.75)),
    transforms.RandomAffine(degrees=26, translate=(0.19, 0.19), scale=(1.19, 1.32), shear=(1.46, 8.4)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
