import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.54), transforms.GaussianBlur(kernel_size=5, sigma=(0.66, 1.57)), transforms.Pad(padding=1, fill=(221, 215, 14), padding_mode='constant'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])