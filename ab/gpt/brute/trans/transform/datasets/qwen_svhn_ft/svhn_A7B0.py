import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomAutocontrast(p=0.23), transforms.RandomEqualize(p=0.63), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])