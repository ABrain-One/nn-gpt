import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.7, 0.87), ratio=(1.13, 2.99)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomGrayscale(p=0.43),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
