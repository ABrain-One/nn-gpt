import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(226, 244, 175), padding_mode='symmetric'), transforms.RandomPosterize(bits=5, p=0.44), transforms.RandomRotation(degrees=8), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])