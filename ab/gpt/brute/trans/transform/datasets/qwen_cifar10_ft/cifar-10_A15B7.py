import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(226, 56, 184), padding_mode='constant'), transforms.CenterCrop(size=32), transforms.RandomVerticalFlip(p=0.81), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])