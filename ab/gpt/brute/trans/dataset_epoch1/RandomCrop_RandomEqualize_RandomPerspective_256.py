import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=24),
    transforms.RandomEqualize(p=0.22),
    transforms.RandomPerspective(distortion_scale=0.23, p=0.85),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
