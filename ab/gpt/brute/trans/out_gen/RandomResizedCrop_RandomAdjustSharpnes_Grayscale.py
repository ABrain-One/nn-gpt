import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.81), ratio=(0.96, 1.66)),
    transforms.RandomAdjustSharpness(sharpness_factor=1.38, p=0.75),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
