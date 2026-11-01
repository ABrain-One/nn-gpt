import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(109, 242, 213), padding_mode='constant'),
    transforms.RandomPerspective(distortion_scale=0.11, p=0.4),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
