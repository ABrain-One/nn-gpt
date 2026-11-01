import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.69, 0.89), ratio=(1.28, 2.19)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(brightness=0.85, contrast=0.91, saturation=1.07, hue=0.07),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
