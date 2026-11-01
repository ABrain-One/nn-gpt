import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=25),
    transforms.RandomAffine(degrees=13, translate=(0.05, 0.04), scale=(0.82, 1.43), shear=(3.88, 9.08)),
    transforms.RandomVerticalFlip(p=0.12),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
