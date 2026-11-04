import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.86), transforms.RandomAdjustSharpness(sharpness_factor=1.73, p=0.89), transforms.RandomGrayscale(p=0.74), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])