import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.55),
    transforms.RandomPerspective(distortion_scale=0.22, p=0.43),
    transforms.RandomPosterize(bits=7, p=0.62),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
