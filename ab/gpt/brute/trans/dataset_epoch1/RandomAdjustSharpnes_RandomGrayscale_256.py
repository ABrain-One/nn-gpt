import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomAdjustSharpness(sharpness_factor=1.24, p=0.35),
    transforms.RandomGrayscale(p=0.78),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
