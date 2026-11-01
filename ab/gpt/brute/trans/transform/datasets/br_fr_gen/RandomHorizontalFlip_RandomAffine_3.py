import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.55),
    transforms.RandomAffine(degrees=21, translate=(0.13, 0.1), scale=(1.01, 1.85), shear=(1.02, 5.46)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
