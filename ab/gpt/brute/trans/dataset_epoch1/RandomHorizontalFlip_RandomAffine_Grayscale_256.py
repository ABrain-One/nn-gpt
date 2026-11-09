import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.47),
    transforms.RandomAffine(degrees=12, translate=(0.12, 0.15), scale=(1.11, 1.76), shear=(4.14, 6.23)),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
