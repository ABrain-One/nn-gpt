import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([transforms.Pad(padding=0, fill=(217, 4, 28), padding_mode='edge'), transforms.CenterCrop(size=32), transforms.RandomEqualize(p=0.42), transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize(*norm)])