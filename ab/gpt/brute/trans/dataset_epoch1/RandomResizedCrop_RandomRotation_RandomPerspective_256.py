import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.8, 0.9), ratio=(1.29, 1.56)),
    transforms.RandomRotation(degrees=23),
    transforms.RandomPerspective(distortion_scale=0.24, p=0.66),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
