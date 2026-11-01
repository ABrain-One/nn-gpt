import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(204, 194, 74), padding_mode='symmetric'), transforms.Grayscale(num_output_channels=3), transforms.RandomAutocontrast(p=0.4), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])