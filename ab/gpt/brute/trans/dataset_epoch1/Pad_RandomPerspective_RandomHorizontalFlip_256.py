import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(160, 71, 110), padding_mode='reflect'),
    transforms.RandomPerspective(distortion_scale=0.27, p=0.13),
    transforms.RandomHorizontalFlip(p=0.12),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
