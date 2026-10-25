import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.11, contrast=0.96, saturation=1.02, hue=0.08),
    transforms.RandomAffine(degrees=11, translate=(0.07, 0.02), scale=(0.97, 1.65), shear=(4.72, 7.16)),
    transforms.CenterCrop(size=32),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
