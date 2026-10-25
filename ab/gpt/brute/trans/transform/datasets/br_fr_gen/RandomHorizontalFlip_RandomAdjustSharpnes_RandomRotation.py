import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.39),
    transforms.RandomAdjustSharpness(sharpness_factor=1.31, p=0.47),
    transforms.RandomRotation(degrees=26),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
