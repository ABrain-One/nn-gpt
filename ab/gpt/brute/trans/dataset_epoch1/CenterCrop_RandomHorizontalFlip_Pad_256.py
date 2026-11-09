import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=31),
    transforms.RandomHorizontalFlip(p=0.57),
    transforms.Pad(padding=2, fill=(111, 175, 144), padding_mode='edge'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
