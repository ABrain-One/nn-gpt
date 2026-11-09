import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.66),
    transforms.RandomAdjustSharpness(sharpness_factor=1.24, p=0.65),
    transforms.RandomGrayscale(p=0.78),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])