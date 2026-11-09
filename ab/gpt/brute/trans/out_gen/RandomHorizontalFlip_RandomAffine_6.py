import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomAffine(degrees=0, translate=(0.13, 0.03), scale=(0.91, 1.52), shear=(3.6, 6.9)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
