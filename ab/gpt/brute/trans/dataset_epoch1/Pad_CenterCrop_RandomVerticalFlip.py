import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(127, 168, 105), padding_mode='reflect'),
    transforms.CenterCrop(size=28),
    transforms.RandomVerticalFlip(p=0.61),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
