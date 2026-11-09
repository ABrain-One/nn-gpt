import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=6, translate=(0.13, 0.11), scale=(0.91, 1.4), shear=(1.5, 6.15)),
    transforms.CenterCrop(size=24),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
