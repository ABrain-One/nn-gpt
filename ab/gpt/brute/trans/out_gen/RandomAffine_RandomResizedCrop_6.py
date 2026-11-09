import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=28, translate=(0.07, 0.12), scale=(0.81, 1.42), shear=(3.83, 7.11)),
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.81), ratio=(0.91, 2.99)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
