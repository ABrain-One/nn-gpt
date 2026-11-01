import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=3, fill=(77, 31, 236), padding_mode='constant'), transforms.RandomEqualize(p=0.53), transforms.RandomGrayscale(p=0.69), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])