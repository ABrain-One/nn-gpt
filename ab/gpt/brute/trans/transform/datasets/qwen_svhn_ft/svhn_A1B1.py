import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(190, 113, 167), padding_mode='reflect'), transforms.Grayscale(num_output_channels=3), transforms.RandomAutocontrast(p=0.57), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])