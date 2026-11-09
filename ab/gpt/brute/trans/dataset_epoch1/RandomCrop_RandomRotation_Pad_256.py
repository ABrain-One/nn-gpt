import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=27),
    transforms.RandomRotation(degrees=0),
    transforms.Pad(padding=1, fill=(244, 172, 229), padding_mode='edge'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
