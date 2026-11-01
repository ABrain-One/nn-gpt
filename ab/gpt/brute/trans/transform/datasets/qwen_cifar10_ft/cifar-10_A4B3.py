import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=5, fill=(122, 135, 239), padding_mode='reflect'), transforms.GaussianBlur(kernel_size=3, sigma=(0.52, 1.84)), transforms.CenterCrop(size=26), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])