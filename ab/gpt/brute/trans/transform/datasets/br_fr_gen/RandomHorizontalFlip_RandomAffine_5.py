import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.8),
    transforms.RandomAffine(degrees=9, translate=(0.07, 0.18), scale=(1.11, 1.64), shear=(2.41, 8.06)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
