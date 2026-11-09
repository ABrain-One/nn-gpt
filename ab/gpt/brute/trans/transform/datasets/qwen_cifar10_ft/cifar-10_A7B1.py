import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=0, fill=(254, 132, 98), padding_mode='constant'), transforms.RandomInvert(p=0.41), transforms.CenterCrop(size=30), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])