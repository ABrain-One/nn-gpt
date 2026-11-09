import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.05, contrast=1.06, saturation=0.82, hue=0.06),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomPosterize(bits=8, p=0.17),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
