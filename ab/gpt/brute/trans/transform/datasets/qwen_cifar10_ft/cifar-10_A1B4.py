import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=5, fill=(109, 19, 19), padding_mode='reflect'), transforms.Grayscale(num_output_channels=3), transforms.CenterCrop(size=30), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])