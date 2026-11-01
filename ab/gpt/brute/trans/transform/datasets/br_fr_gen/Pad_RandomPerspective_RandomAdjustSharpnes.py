import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(88, 69, 140), padding_mode='constant'),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.77),
    transforms.RandomAdjustSharpness(sharpness_factor=1.61, p=0.38),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
