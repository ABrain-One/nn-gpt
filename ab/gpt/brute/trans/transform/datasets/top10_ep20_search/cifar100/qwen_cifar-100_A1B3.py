import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=32), transforms.Pad(padding=2, fill=(12, 6, 132), padding_mode='symmetric'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])