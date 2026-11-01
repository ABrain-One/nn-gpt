import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=32), transforms.RandomVerticalFlip(p=0.55), transforms.Pad(padding=0, fill=(19, 120, 181), padding_mode='constant'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])