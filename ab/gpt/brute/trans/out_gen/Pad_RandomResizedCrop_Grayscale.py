import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(248, 251, 110), padding_mode='reflect'),
    transforms.RandomResizedCrop(size=32, scale=(0.73, 0.82), ratio=(0.79, 2.04)),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
