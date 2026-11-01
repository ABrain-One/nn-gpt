import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.CenterCrop(size=30), transforms.RandomAutocontrast(p=0.73), transforms.Pad(padding=0, fill=(226, 144, 51), padding_mode='edge'), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])