import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=31),
    transforms.RandomCrop(size=27),
    transforms.RandomPerspective(distortion_scale=0.14, p=0.3),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
