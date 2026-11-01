import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=1, fill=(118, 131, 46), padding_mode='edge'), transforms.RandomEqualize(p=0.88), transforms.RandomRotation(degrees=18), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])