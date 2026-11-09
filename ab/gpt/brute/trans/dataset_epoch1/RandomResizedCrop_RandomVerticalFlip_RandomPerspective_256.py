import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.71, 0.99), ratio=(1.24, 1.88)),
    transforms.RandomVerticalFlip(p=0.85),
    transforms.RandomPerspective(distortion_scale=0.18, p=0.89),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
