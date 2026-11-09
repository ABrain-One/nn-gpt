import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.15, contrast=1.07, saturation=0.95, hue=0.05),
    transforms.RandomAffine(degrees=6, translate=(0.02, 0.01), scale=(1.06, 1.57), shear=(1.96, 9.91)),
    transforms.RandomAdjustSharpness(sharpness_factor=0.87, p=0.15),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
