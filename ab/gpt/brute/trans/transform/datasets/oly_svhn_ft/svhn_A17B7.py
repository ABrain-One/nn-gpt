import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(19, 21, 143), padding_mode='reflect'), transforms.RandomGrayscale(p=0.12), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])