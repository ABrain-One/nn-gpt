import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.88, contrast=1.17, saturation=0.99, hue=0.07),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomGrayscale(p=0.31),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
