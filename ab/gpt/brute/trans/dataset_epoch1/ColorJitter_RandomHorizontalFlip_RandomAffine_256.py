import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.06, contrast=0.9, saturation=1.1, hue=0.01),
    transforms.RandomHorizontalFlip(p=0.59),
    transforms.RandomAffine(degrees=23, translate=(0.08, 0.15), scale=(1.02, 1.42), shear=(4.53, 7.53)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
