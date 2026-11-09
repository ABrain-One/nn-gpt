import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(41, 232, 172), padding_mode='reflect'), transforms.RandomGrayscale(p=0.77), transforms.CenterCrop(size=24), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])