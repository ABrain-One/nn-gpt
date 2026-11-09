import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.61),
    transforms.RandomAffine(degrees=9, translate=(0.13, 0.15), scale=(0.8, 1.64), shear=(2.36, 7.43)),
    transforms.RandomEqualize(p=0.58),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
