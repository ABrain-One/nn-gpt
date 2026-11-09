import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=31),
    transforms.Pad(padding=2, fill=(49, 244, 200), padding_mode='reflect'),
    transforms.RandomEqualize(p=0.68),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
