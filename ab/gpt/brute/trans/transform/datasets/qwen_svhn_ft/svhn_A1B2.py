import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=30), transforms.Grayscale(num_output_channels=3), transforms.RandomResizedCrop(size=32, scale=(0.73, 0.95), ratio=(1.06, 1.34)), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])