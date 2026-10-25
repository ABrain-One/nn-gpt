import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAutocontrast(p=0.68), transforms.CenterCrop(size=32), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])