import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=32), transforms.RandomAdjustSharpness(sharpness_factor=0.84, p=0.18), transforms.Pad(padding=5, fill=(121, 91, 13), padding_mode='reflect'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])