import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(109, 23, 52), padding_mode='reflect'),
    transforms.RandomVerticalFlip(p=0.32),
    transforms.RandomSolarize(threshold=38, p=0.45),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
