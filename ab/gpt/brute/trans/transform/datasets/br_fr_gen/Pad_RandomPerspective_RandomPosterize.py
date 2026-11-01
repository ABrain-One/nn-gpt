import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(77, 70, 223), padding_mode='constant'),
    transforms.RandomPerspective(distortion_scale=0.27, p=0.74),
    transforms.RandomPosterize(bits=7, p=0.86),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
