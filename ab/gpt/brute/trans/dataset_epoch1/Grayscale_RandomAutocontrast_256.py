import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomAutocontrast(p=0.11),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
