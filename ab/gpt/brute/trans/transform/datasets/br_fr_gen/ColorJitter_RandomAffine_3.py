import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.02, contrast=1.0, saturation=1.13, hue=0.01),
    transforms.RandomAffine(degrees=17, translate=(0.0, 0.13), scale=(0.81, 1.75), shear=(2.99, 5.39)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
