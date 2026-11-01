import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(165, 82, 112), padding_mode='edge'), transforms.GaussianBlur(kernel_size=3, sigma=(0.71, 1.57)), transforms.RandomAdjustSharpness(sharpness_factor=1.75, p=0.52), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])