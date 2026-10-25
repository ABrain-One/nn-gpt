import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.89, contrast=1.1, saturation=0.89, hue=0.03),
    transforms.CenterCrop(size=27),
    transforms.RandomAffine(degrees=30, translate=(0.19, 0.16), scale=(0.89, 2.0), shear=(4.59, 6.87)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
