import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(22, 152, 93), padding_mode='reflect'), transforms.RandomVerticalFlip(p=0.48), transforms.RandomPosterize(bits=6, p=0.56), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])