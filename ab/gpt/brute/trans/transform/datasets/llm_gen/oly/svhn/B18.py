import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(32), transforms.RandomGrayscale(p=0.75), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.1), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])