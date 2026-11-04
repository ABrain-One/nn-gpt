import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.97), ratio=(0.76, 2.98)),
    transforms.RandomPerspective(distortion_scale=0.19, p=0.55),
    transforms.RandomRotation(degrees=24),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
