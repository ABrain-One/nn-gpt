import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(243, 7, 201), padding_mode='edge'), transforms.RandomAutocontrast(p=0.62), transforms.CenterCrop(size=29), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])