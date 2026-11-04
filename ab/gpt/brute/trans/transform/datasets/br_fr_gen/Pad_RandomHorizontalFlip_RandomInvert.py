import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(88, 174, 42), padding_mode='constant'),
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomInvert(p=0.49),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
