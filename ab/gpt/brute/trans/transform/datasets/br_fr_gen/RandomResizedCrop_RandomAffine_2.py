import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.94), ratio=(0.78, 2.58)),
    transforms.RandomAffine(degrees=13, translate=(0.06, 0.1), scale=(0.96, 1.48), shear=(2.21, 8.35)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
