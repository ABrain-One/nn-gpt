import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(7, 225, 223), padding_mode='symmetric'), transforms.GaussianBlur(kernel_size=3, sigma=(0.74, 1.98)), transforms.RandomAdjustSharpness(sharpness_factor=1.73, p=0.62), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])