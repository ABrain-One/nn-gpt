import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(98, 132, 235), padding_mode='edge'), transforms.RandomAdjustSharpness(sharpness_factor=1.81, p=0.53), transforms.RandomVerticalFlip(p=0.86), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])