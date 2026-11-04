import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(138, 64, 46), padding_mode='reflect'), transforms.GaussianBlur(kernel_size=3, sigma=(0.99, 1.66)), transforms.RandomAdjustSharpness(sharpness_factor=1.65, p=0.21), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])