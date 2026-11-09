import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.99), ratio=(1.09, 1.82)),
    transforms.CenterCrop(size=26),
    transforms.RandomRotation(degrees=11),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
