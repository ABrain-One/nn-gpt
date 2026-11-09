import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.64, 0.9), ratio=(1.23, 2.04)),
    transforms.RandomInvert(p=0.76),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
