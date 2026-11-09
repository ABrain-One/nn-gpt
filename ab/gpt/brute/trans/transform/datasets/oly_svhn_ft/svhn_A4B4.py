import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.RandomAutocontrast(p=0.71), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.2726, 0.2633, 0.2603], std=[0.2459, 0.2462, 0.2451])])