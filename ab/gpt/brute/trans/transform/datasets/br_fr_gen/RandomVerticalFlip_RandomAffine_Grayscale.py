import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomAffine(degrees=15, translate=(0.13, 0.03), scale=(1.09, 1.37), shear=(2.79, 8.96)),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
