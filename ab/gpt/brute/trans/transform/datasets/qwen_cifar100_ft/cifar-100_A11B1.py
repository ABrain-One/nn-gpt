import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(226, 244, 175), padding_mode='edge'), transforms.GaussianBlur(kernel_size=3, sigma=(0.85, 1.66)), transforms.RandomPosterize(bits=4, p=0.17), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])