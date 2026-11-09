import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(69, 161, 251), padding_mode='reflect'),
    transforms.RandomGrayscale(p=0.85),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
