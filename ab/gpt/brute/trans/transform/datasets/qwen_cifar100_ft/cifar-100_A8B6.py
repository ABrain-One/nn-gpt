import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(72, 41, 232), padding_mode='edge'), transforms.RandomVerticalFlip(p=0.64), transforms.RandomAutocontrast(p=0.76), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])