import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.9, contrast=1.15, saturation=0.84, hue=0.09),
    transforms.Pad(padding=2, fill=(111, 235, 143), padding_mode='constant'),
    transforms.RandomSolarize(threshold=32, p=0.73),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
