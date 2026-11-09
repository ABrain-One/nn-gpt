import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.32),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomAffine(degrees=5, translate=(0.11, 0.16), scale=(0.9, 1.72), shear=(4.05, 6.45)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
