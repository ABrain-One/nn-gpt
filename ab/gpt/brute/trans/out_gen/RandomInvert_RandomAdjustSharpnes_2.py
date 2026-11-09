import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomInvert(p=0.2),
    transforms.RandomAdjustSharpness(sharpness_factor=0.99, p=0.68),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
