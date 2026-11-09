import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.32), transforms.RandomGrayscale(p=0.73), transforms.RandomRotation(degrees=11), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])