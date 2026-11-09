import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.7),
    transforms.RandomRotation(degrees=29),
    transforms.RandomAdjustSharpness(sharpness_factor=1.82, p=0.23),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
