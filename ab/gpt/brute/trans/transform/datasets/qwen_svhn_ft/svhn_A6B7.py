import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=5, fill=(77, 31, 243), padding_mode='edge'), transforms.RandomEqualize(p=0.66), transforms.RandomAutocontrast(p=0.75), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])