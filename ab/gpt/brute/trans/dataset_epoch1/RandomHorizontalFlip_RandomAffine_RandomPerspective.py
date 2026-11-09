import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.86),
    transforms.RandomAffine(degrees=14, translate=(0.03, 0.07), scale=(1.01, 1.25), shear=(1.71, 9.05)),
    transforms.RandomPerspective(distortion_scale=0.11, p=0.28),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
