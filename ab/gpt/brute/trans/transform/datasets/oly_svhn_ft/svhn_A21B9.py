import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=4, fill=(142, 21, 162), padding_mode='edge'), transforms.RandomEqualize(p=0.87), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])