import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.79, 0.83), ratio=(1.32, 2.78)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(degrees=15),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
