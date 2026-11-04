import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.27), transforms.RandomRotation(degrees=7), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(*norm)])