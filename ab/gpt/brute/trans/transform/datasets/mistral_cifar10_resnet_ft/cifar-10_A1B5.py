import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(135, 206, 230), padding_mode='constant'), transforms.RandomAutocontrast(p=0.74), transforms.RandomPosterize(bits=6, p=0.63), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])