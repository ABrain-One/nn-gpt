import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.1, p=0.62),
    transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(1.02, 1.92), shear=(0.89, 7.02)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
