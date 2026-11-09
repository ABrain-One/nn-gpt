import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomRotation(degrees=18),
    transforms.RandomAdjustSharpness(sharpness_factor=0.7, p=0.71),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
