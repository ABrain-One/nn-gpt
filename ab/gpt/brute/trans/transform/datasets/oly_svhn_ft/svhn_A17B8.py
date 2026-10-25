import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomResizedCrop(size=32, scale=(0.76, 0.87), ratio=(1.18, 2.83)), transforms.Pad(padding=1, fill=(148, 149, 171), padding_mode='reflect'), transforms.Grayscale(num_output_channels=3), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])