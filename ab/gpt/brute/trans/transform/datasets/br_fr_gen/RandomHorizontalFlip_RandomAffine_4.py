import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.68),
    transforms.RandomAffine(degrees=18, translate=(0.03, 0.13), scale=(0.88, 1.69), shear=(3.24, 9.03)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
