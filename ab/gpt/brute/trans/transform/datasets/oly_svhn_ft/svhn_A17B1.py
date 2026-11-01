import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=31), transforms.Pad(padding=2, fill=(174, 40, 161), padding_mode='reflect'), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])