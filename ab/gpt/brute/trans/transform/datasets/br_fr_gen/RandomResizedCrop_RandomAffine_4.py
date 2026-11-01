import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.91), ratio=(0.84, 1.98)),
    transforms.RandomAffine(degrees=3, translate=(0.18, 0.14), scale=(1.15, 1.3), shear=(2.99, 8.54)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
