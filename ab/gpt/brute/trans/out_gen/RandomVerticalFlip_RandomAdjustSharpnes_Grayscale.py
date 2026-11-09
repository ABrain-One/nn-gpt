import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.87),
    transforms.RandomAdjustSharpness(sharpness_factor=1.25, p=0.24),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
