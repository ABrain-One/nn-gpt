import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=26),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomAffine(degrees=30, translate=(0.01, 0.05), scale=(0.86, 1.57), shear=(1.92, 7.63)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
