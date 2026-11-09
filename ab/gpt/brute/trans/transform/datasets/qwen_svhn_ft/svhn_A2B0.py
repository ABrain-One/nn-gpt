import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=0, fill=(127, 255, 190), padding_mode='reflect'), transforms.RandomInvert(p=0.26), transforms.RandomGrayscale(p=0.58), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])