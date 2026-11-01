import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.RandomGrayscale(p=0.89), transforms.Pad(padding=5, fill=(45, 134, 255), padding_mode='reflect'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])