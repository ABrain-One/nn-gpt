import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=25),
    transforms.CenterCrop(size=30),
    transforms.Pad(padding=0, fill=(47, 173, 147), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
