import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(3, 247, 127), padding_mode='edge'), transforms.RandomGrayscale(p=0.85), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])