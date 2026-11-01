import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=27),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomAffine(degrees=15, translate=(0.12, 0.17), scale=(0.94, 1.37), shear=(1.33, 8.79)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
