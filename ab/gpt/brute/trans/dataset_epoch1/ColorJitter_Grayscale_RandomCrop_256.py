import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.18, contrast=0.87, saturation=0.87, hue=0.06),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomCrop(size=27),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
