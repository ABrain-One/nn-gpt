import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.57),
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.92), ratio=(1.25, 2.3)),
    transforms.RandomAffine(degrees=11, translate=(0.13, 0.11), scale=(0.92, 1.51), shear=(2.83, 7.94)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
