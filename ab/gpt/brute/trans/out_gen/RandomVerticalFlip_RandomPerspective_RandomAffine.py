import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.42),
    transforms.RandomPerspective(distortion_scale=0.24, p=0.72),
    transforms.RandomAffine(degrees=25, translate=(0.03, 0.19), scale=(1.02, 1.58), shear=(1.92, 5.2)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
