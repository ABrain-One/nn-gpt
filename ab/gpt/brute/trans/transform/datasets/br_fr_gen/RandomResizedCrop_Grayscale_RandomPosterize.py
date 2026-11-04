import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.85), ratio=(0.82, 1.66)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomPosterize(bits=7, p=0.36),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
