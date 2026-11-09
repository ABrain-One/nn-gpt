import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(200, 32, 165), padding_mode='edge'),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomGrayscale(p=0.87),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
