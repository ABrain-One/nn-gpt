import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=5, fill=(42, 103, 19), padding_mode='constant'), transforms.RandomPosterize(bits=6, p=0.17), transforms.RandomVerticalFlip(p=0.65), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])