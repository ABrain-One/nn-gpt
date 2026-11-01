import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.01, 0.14), scale=(1.02, 1.23), shear=(0.85, 8.63)),
    transforms.RandomPerspective(distortion_scale=0.11, p=0.64),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
