import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.04, contrast=0.8, saturation=1.09, hue=0.06),
    transforms.RandomVerticalFlip(p=0.11),
    transforms.RandomAffine(degrees=18, translate=(0.09, 0.06), scale=(0.83, 1.55), shear=(1.84, 9.04)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
