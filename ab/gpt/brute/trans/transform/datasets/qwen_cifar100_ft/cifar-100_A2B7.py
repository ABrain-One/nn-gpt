import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=25), transforms.RandomRotation(degrees=8), transforms.Pad(padding=2, fill=(132, 136, 245), padding_mode='edge'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])