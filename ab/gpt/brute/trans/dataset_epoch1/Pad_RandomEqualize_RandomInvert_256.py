import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(101, 218, 174), padding_mode='reflect'),
    transforms.RandomEqualize(p=0.84),
    transforms.RandomInvert(p=0.14),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
