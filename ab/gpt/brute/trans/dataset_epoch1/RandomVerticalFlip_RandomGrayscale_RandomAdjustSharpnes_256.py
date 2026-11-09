import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.52),
    transforms.RandomGrayscale(p=0.34),
    transforms.RandomAdjustSharpness(sharpness_factor=1.63, p=0.12),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
