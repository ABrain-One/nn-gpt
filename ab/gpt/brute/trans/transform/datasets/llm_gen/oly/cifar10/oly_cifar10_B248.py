import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomInvert(p=0.23), transforms.Grayscale(num_output_channels=3), transforms.Resize(64), transforms.RandomHorizontalFlip(p=0.73), transforms.RandomRotation(13), transforms.ToTensor(), transforms.Normalize(*norm)])