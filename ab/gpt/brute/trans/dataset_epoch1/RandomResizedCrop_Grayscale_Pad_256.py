import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.76, 0.88), ratio=(1.23, 2.01)),
    transforms.Grayscale(num_output_channels=3),
    transforms.Pad(padding=0, fill=(175, 190, 250), padding_mode='reflect'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
