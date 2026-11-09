import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(102, 246, 227), padding_mode='symmetric'), transforms.GaussianBlur(kernel_size=5, sigma=(0.78, 1.27)), transforms.RandomRotation(degrees=14), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])