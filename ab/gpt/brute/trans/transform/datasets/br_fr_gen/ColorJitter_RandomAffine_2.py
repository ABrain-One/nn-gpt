import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.83, contrast=0.81, saturation=1.17, hue=0.09),
    transforms.RandomAffine(degrees=5, translate=(0.15, 0.17), scale=(1.04, 1.79), shear=(1.99, 7.5)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
