import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.43),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomPerspective(distortion_scale=0.13, p=0.57),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
