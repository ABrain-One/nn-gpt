import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.23, p=0.8),
    transforms.RandomAffine(degrees=14, translate=(0.01, 0.19), scale=(0.94, 1.87), shear=(4.09, 6.48)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
