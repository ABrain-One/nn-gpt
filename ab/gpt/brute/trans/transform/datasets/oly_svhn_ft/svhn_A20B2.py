import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(142, 0, 127), padding_mode='reflect'), transforms.GaussianBlur(kernel_size=3, sigma=(0.11, 1.9)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])