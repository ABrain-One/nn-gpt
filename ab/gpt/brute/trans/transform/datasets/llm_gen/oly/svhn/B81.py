import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomRotation(degrees=13), transforms.RandomHorizontalFlip(p=0.54), transforms.RandomAutocontrast(p=0.76), transforms.RandomInvert(p=0.64), transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(*norm)])