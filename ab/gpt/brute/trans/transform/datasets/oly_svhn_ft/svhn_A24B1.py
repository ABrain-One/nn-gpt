import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(149, 238, 150), padding_mode='symmetric'), transforms.RandomGrayscale(p=0.16), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.46, 0.44, 0.48], std=[0.265, 0.2668, 0.262])])