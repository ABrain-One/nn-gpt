import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.89, contrast=1.04, saturation=0.94, hue=0.03),
    transforms.RandomAffine(degrees=29, translate=(0.02, 0.02), scale=(1.14, 1.59), shear=(3.47, 5.56)),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
