import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.42),
    transforms.RandomAutocontrast(p=0.5),
    transforms.RandomAffine(degrees=5, translate=(0.15, 0.19), scale=(0.91, 1.96), shear=(3.03, 6.86)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
