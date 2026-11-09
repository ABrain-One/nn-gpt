import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.62),
    transforms.RandomAffine(degrees=3, translate=(0.11, 0.11), scale=(1.15, 1.34), shear=(2.88, 7.09)),
    transforms.ColorJitter(brightness=0.93, contrast=1.02, saturation=0.82, hue=0.1),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
