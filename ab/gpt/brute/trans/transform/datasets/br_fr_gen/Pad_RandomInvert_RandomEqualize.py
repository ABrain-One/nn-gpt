import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(223, 8, 198), padding_mode='constant'),
    transforms.RandomInvert(p=0.35),
    transforms.RandomEqualize(p=0.18),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
