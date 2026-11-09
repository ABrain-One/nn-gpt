import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomGrayscale(p=0.64),
    transforms.RandomAdjustSharpness(sharpness_factor=0.69, p=0.74),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
