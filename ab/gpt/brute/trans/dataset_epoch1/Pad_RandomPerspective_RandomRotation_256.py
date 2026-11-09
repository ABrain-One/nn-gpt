import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(84, 103, 10), padding_mode='edge'),
    transforms.RandomPerspective(distortion_scale=0.27, p=0.16),
    transforms.RandomRotation(degrees=11),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
