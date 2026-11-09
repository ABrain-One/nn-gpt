import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=1, translate=(0.07, 0.03), scale=(0.97, 1.73), shear=(0.7, 7.47)),
    transforms.ColorJitter(brightness=1.18, contrast=1.03, saturation=1.17, hue=0.02),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
