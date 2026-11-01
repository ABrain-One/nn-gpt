import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=7, translate=(0.08, 0.16), scale=(0.96, 1.78), shear=(1.21, 8.41)),
    transforms.RandomHorizontalFlip(p=0.42),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
