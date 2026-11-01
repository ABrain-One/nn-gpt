import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(166, 82, 183), padding_mode='reflect'), transforms.RandomGrayscale(p=0.26), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])