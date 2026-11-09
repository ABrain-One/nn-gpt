import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.78),
    transforms.RandomCrop(size=24),
    transforms.RandomAffine(degrees=24, translate=(0.01, 0.2), scale=(0.95, 1.81), shear=(1.25, 8.88)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
