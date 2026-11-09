import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.78),
    transforms.RandomSolarize(threshold=199, p=0.28),
    transforms.Pad(padding=3, fill=(225, 118, 226), padding_mode='reflect'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
