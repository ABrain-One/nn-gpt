import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.18),
    transforms.RandomInvert(p=0.67),
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.15), scale=(0.83, 1.77), shear=(1.17, 6.15)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
