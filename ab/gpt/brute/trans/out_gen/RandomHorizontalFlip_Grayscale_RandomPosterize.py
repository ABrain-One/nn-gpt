import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.83),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomPosterize(bits=6, p=0.13),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
