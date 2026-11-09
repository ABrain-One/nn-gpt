import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(152, 138, 164), padding_mode='reflect'),
    transforms.RandomVerticalFlip(p=0.4),
    transforms.CenterCrop(size=29),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
