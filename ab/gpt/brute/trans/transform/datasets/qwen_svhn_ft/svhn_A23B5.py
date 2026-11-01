import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=0, fill=(245, 165, 98), padding_mode='edge'), transforms.RandomAutocontrast(p=0.62), transforms.Grayscale(num_output_channels=3), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])