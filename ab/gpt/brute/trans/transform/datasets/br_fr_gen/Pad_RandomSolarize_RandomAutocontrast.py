import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(173, 195, 128), padding_mode='edge'),
    transforms.RandomSolarize(threshold=57, p=0.11),
    transforms.RandomAutocontrast(p=0.73),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
