import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=2, fill=(135, 90, 253), padding_mode='reflect'), transforms.RandomEqualize(p=0.36), transforms.RandomInvert(p=0.45), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])