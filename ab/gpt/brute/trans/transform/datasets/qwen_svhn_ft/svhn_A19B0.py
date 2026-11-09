import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=26), transforms.RandomGrayscale(p=0.51), transforms.Pad(padding=1, fill=(203, 14, 33), padding_mode='reflect'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])