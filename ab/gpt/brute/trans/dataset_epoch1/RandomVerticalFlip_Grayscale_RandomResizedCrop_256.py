import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.78),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(size=32, scale=(0.67, 0.9), ratio=(0.86, 1.53)),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
