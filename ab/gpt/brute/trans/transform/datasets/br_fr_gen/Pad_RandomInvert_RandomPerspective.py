import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(237, 146, 201), padding_mode='edge'),
    transforms.RandomInvert(p=0.89),
    transforms.RandomPerspective(distortion_scale=0.14, p=0.4),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
