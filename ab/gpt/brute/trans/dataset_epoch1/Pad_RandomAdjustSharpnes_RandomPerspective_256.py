import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(87, 15, 83), padding_mode='constant'),
    transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=0.37),
    transforms.RandomPerspective(distortion_scale=0.28, p=0.5),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
