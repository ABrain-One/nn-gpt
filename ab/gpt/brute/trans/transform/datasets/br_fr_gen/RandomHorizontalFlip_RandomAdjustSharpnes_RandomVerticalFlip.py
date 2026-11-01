import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.59),
    transforms.RandomAdjustSharpness(sharpness_factor=1.82, p=0.73),
    transforms.RandomVerticalFlip(p=0.82),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
