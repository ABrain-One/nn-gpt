import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.23),
    transforms.RandomCrop(size=24),
    transforms.RandomAdjustSharpness(sharpness_factor=1.38, p=0.76),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
