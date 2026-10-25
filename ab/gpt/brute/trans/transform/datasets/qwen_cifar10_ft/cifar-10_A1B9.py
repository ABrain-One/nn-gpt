import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(41, 61, 212), padding_mode='reflect'), transforms.RandomHorizontalFlip(p=0.63), transforms.GaussianBlur(kernel_size=3, sigma=(0.76, 1.22)), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])