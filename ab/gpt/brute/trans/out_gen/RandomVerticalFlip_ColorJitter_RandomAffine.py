import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.34),
    transforms.ColorJitter(brightness=0.82, contrast=0.84, saturation=1.07, hue=0.0),
    transforms.RandomAffine(degrees=26, translate=(0.13, 0.14), scale=(1.12, 1.96), shear=(4.12, 7.94)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
