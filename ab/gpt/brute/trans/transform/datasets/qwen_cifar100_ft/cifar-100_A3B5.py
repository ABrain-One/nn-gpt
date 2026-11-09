import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.14), transforms.Pad(padding=4, fill=(215, 44, 99), padding_mode='constant'), transforms.GaussianBlur(kernel_size=3, sigma=(0.39, 1.96)), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])