import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=28),
    transforms.Grayscale(num_output_channels=3),
    transforms.ColorJitter(brightness=0.92, contrast=0.91, saturation=1.09, hue=0.07),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
