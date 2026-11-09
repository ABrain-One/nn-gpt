import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=26), transforms.RandomVerticalFlip(p=0.85), transforms.Pad(padding=3, fill=(7, 24, 235), padding_mode='constant'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])