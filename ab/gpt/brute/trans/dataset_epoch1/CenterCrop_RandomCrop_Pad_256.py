import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.RandomCrop(size=29),
    transforms.Pad(padding=5, fill=(90, 42, 134), padding_mode='constant'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
