import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.4), transforms.RandomGrayscale(p=0.5), transforms.Grayscale(num_output_channels=3), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])