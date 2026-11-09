import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.55),
    transforms.RandomAffine(degrees=9, translate=(0.11, 0.07), scale=(0.98, 1.74), shear=(1.98, 9.36)),
    transforms.RandomAutocontrast(p=0.61),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
