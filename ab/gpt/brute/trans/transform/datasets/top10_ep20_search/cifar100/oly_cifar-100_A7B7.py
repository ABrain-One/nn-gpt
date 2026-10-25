import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=5, fill=(152, 103, 42), padding_mode='constant'), transforms.RandomPosterize(bits=4, p=0.35), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])