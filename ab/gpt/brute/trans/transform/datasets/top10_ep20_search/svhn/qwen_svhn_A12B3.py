import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAdjustSharpness(sharpness_factor=0.88, p=0.14), transforms.RandomAutocontrast(p=0.62), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])