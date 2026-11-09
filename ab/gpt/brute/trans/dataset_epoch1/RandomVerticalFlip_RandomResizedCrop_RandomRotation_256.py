import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.79),
    transforms.RandomResizedCrop(size=32, scale=(0.55, 0.95), ratio=(0.92, 2.19)),
    transforms.RandomRotation(degrees=5),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
