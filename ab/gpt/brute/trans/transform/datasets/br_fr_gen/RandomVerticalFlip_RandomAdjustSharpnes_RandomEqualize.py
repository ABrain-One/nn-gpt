import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.6),
    transforms.RandomAdjustSharpness(sharpness_factor=0.79, p=0.8),
    transforms.RandomEqualize(p=0.15),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
