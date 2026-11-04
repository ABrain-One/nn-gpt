import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=0, fill=(254, 162, 90), padding_mode='constant'), transforms.RandomInvert(p=0.41), transforms.RandomPosterize(bits=6, p=0.45), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])