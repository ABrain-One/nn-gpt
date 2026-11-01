import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.14),
    transforms.Pad(padding=1, fill=(138, 104, 226), padding_mode='edge'),
    transforms.RandomEqualize(p=0.62),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
