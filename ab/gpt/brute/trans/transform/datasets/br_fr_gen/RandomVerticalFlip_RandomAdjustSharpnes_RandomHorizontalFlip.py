import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.58),
    transforms.RandomAdjustSharpness(sharpness_factor=1.81, p=0.58),
    transforms.RandomHorizontalFlip(p=0.33),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
