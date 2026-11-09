import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=26, translate=(0.12, 0.03), scale=(1.08, 1.37), shear=(2.6, 5.83)),
    transforms.RandomResizedCrop(size=32, scale=(0.71, 0.81), ratio=(1.12, 1.37)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
