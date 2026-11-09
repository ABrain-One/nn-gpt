import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.53),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomAffine(degrees=4, translate=(0.01, 0.02), scale=(1.05, 1.65), shear=(0.15, 7.04)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
