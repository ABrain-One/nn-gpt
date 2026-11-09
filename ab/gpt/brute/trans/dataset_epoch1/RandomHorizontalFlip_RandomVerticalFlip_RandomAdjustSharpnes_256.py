import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.87),
    transforms.RandomVerticalFlip(p=0.69),
    transforms.RandomAdjustSharpness(sharpness_factor=1.13, p=0.88),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
