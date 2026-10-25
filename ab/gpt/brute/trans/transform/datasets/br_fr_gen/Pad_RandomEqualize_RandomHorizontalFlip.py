import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(40, 25, 43), padding_mode='constant'),
    transforms.RandomEqualize(p=0.63),
    transforms.RandomHorizontalFlip(p=0.82),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
