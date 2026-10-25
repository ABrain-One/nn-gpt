import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.96), ratio=(1.12, 2.16)),
    transforms.RandomAffine(degrees=27, translate=(0.02, 0.17), scale=(0.84, 1.37), shear=(4.99, 5.56)),
    transforms.RandomVerticalFlip(p=0.81),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
