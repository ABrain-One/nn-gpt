import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.74),
    transforms.RandomInvert(p=0.35),
    transforms.RandomAdjustSharpness(sharpness_factor=1.82, p=0.74),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
