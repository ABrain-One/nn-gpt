import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomPosterize(bits=7, p=0.36),
    transforms.Pad(padding=4, fill=(210, 225, 181), padding_mode='constant'),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
