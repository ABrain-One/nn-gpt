import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.12, 0.04), scale=(1.0, 1.8), shear=(1.57, 6.78)),
    transforms.RandomPerspective(distortion_scale=0.18, p=0.87),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
