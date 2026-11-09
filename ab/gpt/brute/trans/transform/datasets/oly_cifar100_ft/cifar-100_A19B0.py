import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomHorizontalFlip(p=0.4), transforms.RandomCrop(size=29), transforms.RandomGrayscale(p=0.32), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])