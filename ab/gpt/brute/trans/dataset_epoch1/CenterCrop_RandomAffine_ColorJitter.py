import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.RandomAffine(degrees=14, translate=(0.02, 0.16), scale=(0.81, 1.75), shear=(1.39, 7.94)),
    transforms.ColorJitter(brightness=1.1, contrast=1.15, saturation=1.02, hue=0.05),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
