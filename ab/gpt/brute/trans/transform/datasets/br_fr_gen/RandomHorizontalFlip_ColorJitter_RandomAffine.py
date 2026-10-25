import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.41),
    transforms.ColorJitter(brightness=1.14, contrast=1.11, saturation=1.17, hue=0.09),
    transforms.RandomAffine(degrees=2, translate=(0.11, 0.12), scale=(1.1, 1.26), shear=(1.33, 9.39)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
