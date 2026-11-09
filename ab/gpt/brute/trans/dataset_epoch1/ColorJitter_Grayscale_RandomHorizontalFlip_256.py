import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.84, contrast=0.81, saturation=0.89, hue=0.0),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.79),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
