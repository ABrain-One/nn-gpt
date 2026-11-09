import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.11, contrast=1.08, saturation=0.94, hue=0.05),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomEqualize(p=0.43),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
