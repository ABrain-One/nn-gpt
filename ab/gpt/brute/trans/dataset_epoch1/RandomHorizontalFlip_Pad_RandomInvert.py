import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.86),
    transforms.Pad(padding=2, fill=(119, 18, 25), padding_mode='constant'),
    transforms.RandomInvert(p=0.73),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
