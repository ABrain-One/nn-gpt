import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(size=30), transforms.RandomAdjustSharpness(sharpness_factor=1.17, p=0.33), transforms.Pad(padding=3, fill=(64, 125, 245), padding_mode='edge'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])