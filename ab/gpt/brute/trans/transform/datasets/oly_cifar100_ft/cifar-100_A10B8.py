import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.36), transforms.RandomGrayscale(p=0.89), transforms.RandomSolarize(threshold=95, p=0.25), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])