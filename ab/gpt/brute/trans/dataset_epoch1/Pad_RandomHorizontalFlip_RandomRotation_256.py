import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(194, 244, 182), padding_mode='edge'),
    transforms.RandomHorizontalFlip(p=0.54),
    transforms.RandomRotation(degrees=13),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
