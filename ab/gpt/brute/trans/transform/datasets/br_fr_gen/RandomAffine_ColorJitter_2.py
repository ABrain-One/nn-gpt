import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=8, translate=(0.11, 0.18), scale=(0.9, 1.24), shear=(0.9, 5.69)),
    transforms.ColorJitter(brightness=1.14, contrast=1.09, saturation=1.17, hue=0.05),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
