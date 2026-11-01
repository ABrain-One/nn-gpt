import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(16, 208, 150), padding_mode='constant'),
    transforms.RandomCrop(size=26),
    transforms.RandomPosterize(bits=8, p=0.2),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
