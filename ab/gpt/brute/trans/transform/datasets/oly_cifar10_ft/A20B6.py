import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(13, 199, 80), padding_mode='edge'),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomAdjustSharpness(sharpness_factor=0.57, p=0.39),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])