import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.21),
    transforms.RandomAutocontrast(p=0.57),
    transforms.RandomAffine(degrees=15, translate=(0.15, 0.1), scale=(1.14, 1.4), shear=(4.42, 9.98)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
