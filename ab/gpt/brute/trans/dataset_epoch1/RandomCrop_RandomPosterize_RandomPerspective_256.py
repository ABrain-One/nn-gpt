import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=24),
    transforms.RandomPosterize(bits=6, p=0.36),
    transforms.RandomPerspective(distortion_scale=0.16, p=0.72),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
