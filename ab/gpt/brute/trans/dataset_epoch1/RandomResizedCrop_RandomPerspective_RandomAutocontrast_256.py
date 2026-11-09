import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.96), ratio=(0.8, 2.65)),
    transforms.RandomPerspective(distortion_scale=0.25, p=0.79),
    transforms.RandomAutocontrast(p=0.13),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
