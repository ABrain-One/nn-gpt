import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.52),
    transforms.RandomPerspective(distortion_scale=0.12, p=0.45),
    transforms.Pad(padding=0, fill=(248, 6, 129), padding_mode='constant'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
