import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=24),
    transforms.RandomVerticalFlip(p=0.83),
    transforms.RandomAdjustSharpness(sharpness_factor=1.6, p=0.17),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
