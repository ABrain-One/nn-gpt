import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomVerticalFlip(p=0.76), transforms.RandomGrayscale(p=0.65), transforms.RandomEqualize(p=0.65), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])