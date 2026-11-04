import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=28), transforms.Pad(padding=3, fill=(255, 103, 171), padding_mode='edge'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])