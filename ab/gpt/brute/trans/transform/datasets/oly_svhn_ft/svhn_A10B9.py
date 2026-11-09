import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(165, 82, 183), padding_mode='symmetric'), transforms.RandomInvert(p=0.13), transforms.RandomPosterize(bits=7, p=0.62), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])