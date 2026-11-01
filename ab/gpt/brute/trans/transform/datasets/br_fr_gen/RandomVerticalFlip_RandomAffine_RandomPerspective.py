import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.54),
    transforms.RandomAffine(degrees=18, translate=(0.15, 0.03), scale=(1.13, 1.46), shear=(2.22, 5.12)),
    transforms.RandomPerspective(distortion_scale=0.16, p=0.8),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
