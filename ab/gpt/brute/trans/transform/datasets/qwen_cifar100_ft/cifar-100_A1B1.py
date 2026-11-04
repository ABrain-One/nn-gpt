import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(184, 76, 220), padding_mode='reflect'), transforms.RandomHorizontalFlip(p=0.18), transforms.RandomEqualize(p=0.78), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])