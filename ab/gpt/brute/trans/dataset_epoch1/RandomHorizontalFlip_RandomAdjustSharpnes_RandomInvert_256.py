import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.RandomAdjustSharpness(sharpness_factor=1.45, p=0.63),
    transforms.RandomInvert(p=0.82),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
