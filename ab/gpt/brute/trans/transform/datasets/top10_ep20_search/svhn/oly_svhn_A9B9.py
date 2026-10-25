import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(160, 71, 110), padding_mode='reflect'), transforms.RandomAutocontrast(p=0.61), transforms.RandomPosterize(bits=7, p=0.67), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])