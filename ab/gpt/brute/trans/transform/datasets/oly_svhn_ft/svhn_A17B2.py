import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(150, 51, 187), padding_mode='reflect'), transforms.RandomAutocontrast(p=0.3), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])