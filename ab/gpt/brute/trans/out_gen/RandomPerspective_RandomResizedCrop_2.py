import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.81), ratio=(1.01, 1.99)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
