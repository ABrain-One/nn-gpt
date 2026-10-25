import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.46),
    transforms.RandomAffine(degrees=25, translate=(0.18, 0.17), scale=(1.12, 1.36), shear=(3.89, 8.6)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
