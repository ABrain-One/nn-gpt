import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.12, contrast=0.96, saturation=0.95, hue=0.08),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomPerspective(distortion_scale=0.29, p=0.76),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
