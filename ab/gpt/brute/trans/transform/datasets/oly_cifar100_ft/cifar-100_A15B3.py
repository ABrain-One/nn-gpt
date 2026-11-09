import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=31), transforms.RandomRotation(degrees=15), transforms.RandomGrayscale(p=0.25), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])