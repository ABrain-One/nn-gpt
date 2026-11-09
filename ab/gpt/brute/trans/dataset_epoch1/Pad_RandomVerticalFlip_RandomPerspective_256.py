import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(114, 134, 1), padding_mode='constant'),
    transforms.RandomVerticalFlip(p=0.16),
    transforms.RandomPerspective(distortion_scale=0.22, p=0.58),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
