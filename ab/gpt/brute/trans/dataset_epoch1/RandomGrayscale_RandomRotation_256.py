import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.18),
    transforms.RandomRotation(degrees=8),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
