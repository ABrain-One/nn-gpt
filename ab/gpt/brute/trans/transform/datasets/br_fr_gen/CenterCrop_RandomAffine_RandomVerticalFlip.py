import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=30),
    transforms.RandomAffine(degrees=29, translate=(0.05, 0.19), scale=(1.0, 1.71), shear=(4.28, 8.79)),
    transforms.RandomVerticalFlip(p=0.55),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
