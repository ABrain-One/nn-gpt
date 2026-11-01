import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=5, fill=(195, 216, 105), padding_mode='reflect'), transforms.GaussianBlur(kernel_size=5, sigma=(0.91, 1.25)), transforms.Grayscale(num_output_channels=3), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])