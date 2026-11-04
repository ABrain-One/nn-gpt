import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(72, 41, 30), padding_mode='reflect'),
    transforms.RandomGrayscale(p=0.57),
    transforms.RandomAffine(degrees=10, translate=(0.0, 0.07), scale=(1.07, 1.89), shear=(1.24, 7.47)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
