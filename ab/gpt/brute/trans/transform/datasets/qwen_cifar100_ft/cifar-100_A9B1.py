import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAutocontrast(p=0.54), transforms.RandomRotation(degrees=3), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])