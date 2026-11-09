import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.78, 0.94), ratio=(1.08, 1.61)),
    transforms.ColorJitter(brightness=1.08, contrast=0.95, saturation=1.11, hue=0.03),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
