import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=5, fill=(24, 145, 20), padding_mode='symmetric'), transforms.GaussianBlur(kernel_size=3, sigma=1.0), transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), shear=5, p=0.5), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])