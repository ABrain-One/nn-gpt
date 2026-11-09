import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.54, 0.8), ratio=(0.86, 1.61)),
    transforms.RandomRotation(degrees=13),
    transforms.RandomAdjustSharpness(sharpness_factor=1.62, p=0.53),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
