import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.42),
    transforms.RandomAffine(degrees=29, translate=(0.15, 0.11), scale=(0.87, 1.39), shear=(3.57, 7.6)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
