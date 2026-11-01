import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.86, contrast=1.03, saturation=0.95, hue=0.09),
    transforms.RandomAffine(degrees=9, translate=(0.06, 0.19), scale=(1.07, 1.74), shear=(0.91, 9.19)),
    transforms.RandomVerticalFlip(p=0.48),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
