import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=30),
    transforms.Pad(padding=2, fill=(234, 40, 187), padding_mode='reflect'),
    transforms.RandomSolarize(threshold=50, p=0.59),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
