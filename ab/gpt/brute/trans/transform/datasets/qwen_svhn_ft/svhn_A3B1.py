import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=0, fill=(135, 75, 252), padding_mode='reflect'), transforms.CenterCrop(size=31), transforms.RandomAdjustSharpness(sharpness_factor=0.72, p=0.89), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])