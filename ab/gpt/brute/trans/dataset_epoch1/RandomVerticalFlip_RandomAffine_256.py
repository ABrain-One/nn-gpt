import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.48),
    transforms.RandomAffine(degrees=2, translate=(0.02, 0.08), scale=(0.87, 1.46), shear=(4.46, 9.37)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
