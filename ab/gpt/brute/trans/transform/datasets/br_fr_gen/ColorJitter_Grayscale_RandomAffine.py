import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.81, contrast=0.83, saturation=1.02, hue=0.07),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomAffine(degrees=23, translate=(0.01, 0.11), scale=(1.13, 1.96), shear=(0.34, 9.23)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
