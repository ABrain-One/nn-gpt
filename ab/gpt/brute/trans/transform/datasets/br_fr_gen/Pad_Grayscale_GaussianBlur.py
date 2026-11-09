import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(100, 119, 138), padding_mode='reflect'),
    transforms.Grayscale(num_output_channels=3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.14, 1.28)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
