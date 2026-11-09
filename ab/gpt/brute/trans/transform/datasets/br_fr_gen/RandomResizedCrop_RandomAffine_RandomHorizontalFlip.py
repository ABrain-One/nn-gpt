import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.63, 0.99), ratio=(1.21, 1.36)),
    transforms.RandomAffine(degrees=26, translate=(0.13, 0.18), scale=(0.96, 1.73), shear=(0.88, 5.82)),
    transforms.RandomHorizontalFlip(p=0.23),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
