import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.RandomVerticalFlip(p=0.83),
    transforms.RandomAffine(degrees=5, translate=(0.15, 0.05), scale=(1.07, 1.43), shear=(4.84, 6.23)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
