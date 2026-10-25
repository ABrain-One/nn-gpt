import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.Pad(padding=1, fill=(255, 198, 142), padding_mode='reflect'),
        transforms.RandomAdjustSharpness(sharpness_factor=0.58, p=0.7),
        transforms.RandomGrayscale(p=0.83),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])