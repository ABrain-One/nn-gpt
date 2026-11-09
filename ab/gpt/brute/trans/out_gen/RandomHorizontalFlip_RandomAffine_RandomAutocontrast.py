import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.78),
    transforms.RandomAffine(degrees=3, translate=(0.14, 0.12), scale=(0.92, 1.2), shear=(3.52, 5.4)),
    transforms.RandomAutocontrast(p=0.53),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
