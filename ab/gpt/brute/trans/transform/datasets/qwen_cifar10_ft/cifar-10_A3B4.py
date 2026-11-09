import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(170, 141, 150), padding_mode='reflect'), transforms.CenterCrop(size=31), transforms.Grayscale(num_output_channels=3), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])