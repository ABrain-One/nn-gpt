import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=5, fill=(245, 19, 22), padding_mode='edge'), transforms.GaussianBlur(kernel_size=5, sigma=(0.15, 1.18)), transforms.RandomEqualize(p=0.56), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])