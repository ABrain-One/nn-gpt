import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(13, 131, 213), padding_mode='reflect'), transforms.RandomAutocontrast(p=0.61), transforms.RandomPosterize(bits=7, p=0.12), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])