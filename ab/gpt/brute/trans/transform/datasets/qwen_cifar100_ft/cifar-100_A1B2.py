import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(47, 161, 185), padding_mode='reflect'), transforms.GaussianBlur(kernel_size=3, sigma=(0.79, 1.56)), transforms.RandomPosterize(bits=7, p=0.16), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])