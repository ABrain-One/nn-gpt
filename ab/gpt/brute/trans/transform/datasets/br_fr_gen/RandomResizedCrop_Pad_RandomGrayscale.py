import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.62, 0.87), ratio=(0.87, 2.66)),
    transforms.Pad(padding=3, fill=(230, 119, 67), padding_mode='reflect'),
    transforms.RandomGrayscale(p=0.79),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
