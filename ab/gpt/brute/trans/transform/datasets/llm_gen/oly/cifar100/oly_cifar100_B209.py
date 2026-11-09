import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.ColorJitter(0.92, 1.07, 0.88, 0.09), transforms.RandomHorizontalFlip(p=0.78), transforms.RandomVerticalFlip(p=0.37), transforms.RandomEqualize(p=0.38), transforms.RandomGrayscale(p=0.64), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])