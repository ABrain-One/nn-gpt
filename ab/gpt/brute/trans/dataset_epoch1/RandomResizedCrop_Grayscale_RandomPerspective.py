import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.58, 0.93), ratio=(1.0, 2.56)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomPerspective(distortion_scale=0.14, p=0.88),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
