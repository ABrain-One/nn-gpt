import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=16, translate=(0.13, 0.03), scale=(0.88, 1.65), shear=(0.93, 8.82)),
    transforms.RandomHorizontalFlip(p=0.42),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
