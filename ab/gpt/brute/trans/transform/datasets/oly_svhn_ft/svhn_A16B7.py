import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=0, fill=(142, 147, 173), padding_mode='reflect'), transforms.GaussianBlur(kernel_size=3, sigma=(0.76, 1.92)), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])