import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.31),
    transforms.RandomVerticalFlip(p=0.87),
    transforms.Pad(padding=0, fill=(91, 232, 21), padding_mode='constant'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
