import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=0, fill=(118, 93, 71), padding_mode='edge'), transforms.RandomEqualize(p=0.53), transforms.CenterCrop(size=32), transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(*norm)])