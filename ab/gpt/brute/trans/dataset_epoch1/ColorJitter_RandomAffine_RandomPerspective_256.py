import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.16, contrast=0.97, saturation=1.19, hue=0.04),
    transforms.RandomAffine(degrees=15, translate=(0.05, 0.08), scale=(1.08, 1.41), shear=(4.04, 5.81)),
    transforms.RandomPerspective(distortion_scale=0.22, p=0.56),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
