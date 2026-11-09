import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.22), transforms.Grayscale(num_output_channels=3), transforms.RandomPerspective(distortion_scale=0.11, p=0.77), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])