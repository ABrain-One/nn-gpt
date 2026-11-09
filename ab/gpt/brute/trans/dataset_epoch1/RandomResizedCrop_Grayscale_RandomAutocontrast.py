import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.57, 0.87), ratio=(0.83, 2.3)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomAutocontrast(p=0.21),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
