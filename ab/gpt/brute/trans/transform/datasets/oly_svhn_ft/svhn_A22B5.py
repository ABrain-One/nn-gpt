import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(164, 118, 222), padding_mode='reflect'), transforms.RandomGrayscale(p=0.57), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])