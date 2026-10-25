import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(198, 7, 128), padding_mode='reflect'), transforms.RandomGrayscale(p=0.64), transforms.RandomEqualize(p=0.16), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])