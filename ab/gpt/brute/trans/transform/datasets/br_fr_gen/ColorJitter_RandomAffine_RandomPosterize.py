import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.13, contrast=1.15, saturation=1.13, hue=0.01),
    transforms.RandomAffine(degrees=8, translate=(0.08, 0.04), scale=(1.09, 1.55), shear=(2.24, 8.57)),
    transforms.RandomPosterize(bits=4, p=0.44),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
