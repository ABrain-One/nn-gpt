import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.71),
    transforms.RandomInvert(p=0.58),
    transforms.RandomAdjustSharpness(sharpness_factor=0.53, p=0.87),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
