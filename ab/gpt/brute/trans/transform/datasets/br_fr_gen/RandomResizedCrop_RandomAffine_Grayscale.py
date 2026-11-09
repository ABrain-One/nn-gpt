import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.66, 0.84), ratio=(0.93, 1.98)),
    transforms.RandomAffine(degrees=26, translate=(0.19, 0.01), scale=(1.06, 1.45), shear=(0.56, 8.87)),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
