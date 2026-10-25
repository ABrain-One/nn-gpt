import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomCrop(size=31), transforms.Pad(padding=0, fill=(100, 100, 100), padding_mode='constant'), transforms.RandomGrayscale(p=0.68), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])