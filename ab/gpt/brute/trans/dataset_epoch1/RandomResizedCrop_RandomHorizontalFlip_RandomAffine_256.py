import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.92), ratio=(0.82, 2.0)),
    transforms.RandomHorizontalFlip(p=0.47),
    transforms.RandomAffine(degrees=13, translate=(0.09, 0.04), scale=(1.12, 1.57), shear=(2.83, 6.36)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
