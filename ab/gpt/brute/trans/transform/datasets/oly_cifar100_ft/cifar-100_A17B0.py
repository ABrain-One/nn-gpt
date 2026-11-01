import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(174, 146, 74), padding_mode='reflect'), transforms.RandomPosterize(bits=6, p=0.85), transforms.GaussianBlur(kernel_size=5, sigma=(0.84, 1.11)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])