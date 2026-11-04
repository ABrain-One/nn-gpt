import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=5, fill=(135, 28, 227), padding_mode='symmetric'), transforms.RandomAdjustSharpness(sharpness_factor=0.56, p=0.61), transforms.GaussianBlur(kernel_size=5, sigma=(0.89, 1.03)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])