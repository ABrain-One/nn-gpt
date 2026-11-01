import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(72, 190, 190), padding_mode='edge'), transforms.CenterCrop(size=31), transforms.GaussianBlur(kernel_size=5, sigma=(0.71, 1.7)), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])