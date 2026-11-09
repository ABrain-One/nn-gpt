import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(221, 184, 119), padding_mode='constant'),
    transforms.RandomSolarize(threshold=171, p=0.28),
    transforms.RandomVerticalFlip(p=0.33),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
