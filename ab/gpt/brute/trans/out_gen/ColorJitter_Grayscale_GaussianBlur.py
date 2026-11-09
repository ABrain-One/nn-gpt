import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.14, contrast=0.96, saturation=0.95, hue=0.01),
    transforms.Grayscale(num_output_channels=3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.72, 1.32)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
