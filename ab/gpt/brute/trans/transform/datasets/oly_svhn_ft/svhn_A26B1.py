import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(161, 8, 109), padding_mode='reflect'), transforms.RandomGrayscale(p=0.79), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])