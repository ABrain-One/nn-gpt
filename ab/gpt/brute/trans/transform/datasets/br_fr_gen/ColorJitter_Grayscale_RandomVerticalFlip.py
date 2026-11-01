import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.19, contrast=0.96, saturation=1.08, hue=0.06),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomVerticalFlip(p=0.49),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
