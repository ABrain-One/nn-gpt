import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=28), transforms.Pad(padding=1, fill=(27, 70, 163), padding_mode='edge'), transforms.RandomAutocontrast(p=0.87), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])