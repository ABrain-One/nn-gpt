import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(28, 19, 78), padding_mode='constant'),
    transforms.RandomAffine(degrees=24, translate=(0.1, 0.18), scale=(1.05, 1.94), shear=(0.37, 7.99)),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
