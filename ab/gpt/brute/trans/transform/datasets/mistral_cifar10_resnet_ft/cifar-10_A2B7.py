import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=0, fill=(148, 202, 171), padding_mode='edge'), transforms.RandomInvert(p=0.89), transforms.RandomAutocontrast(p=0.71), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])