import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.14),
    transforms.RandomHorizontalFlip(p=0.34),
    transforms.RandomAdjustSharpness(sharpness_factor=0.88, p=0.14),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
