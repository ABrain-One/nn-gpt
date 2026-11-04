import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.5), transforms.RandomAutocontrast(p=0.5), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.6392, 0.6439, 0.647], std=[0.29, 0.2941, 0.2891])])