import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.59, 0.89), ratio=(1.3, 2.85)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.4, p=0.8),
    transforms.RandomVerticalFlip(p=0.73),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
