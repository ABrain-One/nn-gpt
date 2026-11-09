import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.0, contrast=1.03, saturation=1.04, hue=0.09),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomInvert(p=0.81),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
