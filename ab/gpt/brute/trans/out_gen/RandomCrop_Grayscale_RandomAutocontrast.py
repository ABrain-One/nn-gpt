import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=26),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomAutocontrast(p=0.62),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
