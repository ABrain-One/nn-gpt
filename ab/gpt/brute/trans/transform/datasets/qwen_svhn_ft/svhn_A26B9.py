import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=0, fill=(198, 245, 56), padding_mode='constant'), transforms.GaussianBlur(kernel_size=3, sigma=(0.52, 1.91)), transforms.Grayscale(num_output_channels=3), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])