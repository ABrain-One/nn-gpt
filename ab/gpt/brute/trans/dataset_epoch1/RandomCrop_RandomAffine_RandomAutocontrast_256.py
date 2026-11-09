import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=26),
    transforms.RandomAffine(degrees=13, translate=(0.14, 0.0), scale=(1.19, 1.62), shear=(2.2, 5.43)),
    transforms.RandomAutocontrast(p=0.75),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
