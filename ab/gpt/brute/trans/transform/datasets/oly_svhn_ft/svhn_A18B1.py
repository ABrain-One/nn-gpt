import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(146, 87, 238), padding_mode='symmetric'), transforms.RandomGrayscale(p=0.1), transforms.RandomInvert(p=0.14), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])