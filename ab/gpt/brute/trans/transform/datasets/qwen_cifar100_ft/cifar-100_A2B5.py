import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(7, 17, 199), padding_mode='constant'), transforms.GaussianBlur(kernel_size=5, sigma=(0.54, 1.55)), transforms.RandomVerticalFlip(p=0.38), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])