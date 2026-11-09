import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.65, 0.93), ratio=(0.97, 1.41)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomVerticalFlip(p=0.64),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
