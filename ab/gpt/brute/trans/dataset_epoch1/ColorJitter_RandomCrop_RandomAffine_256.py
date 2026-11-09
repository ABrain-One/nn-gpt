import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.02, contrast=1.13, saturation=0.89, hue=0.05),
    transforms.RandomCrop(size=31),
    transforms.RandomAffine(degrees=27, translate=(0.16, 0.1), scale=(1.02, 1.96), shear=(1.3, 9.7)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
