import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(25, 240, 211), padding_mode='reflect'),
    transforms.RandomEqualize(p=0.9),
    transforms.RandomPerspective(distortion_scale=0.19, p=0.79),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
