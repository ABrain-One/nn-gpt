import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(250, 118, 64), padding_mode='edge'), transforms.RandomEqualize(p=0.89), transforms.RandomCrop(size=30), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])