import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.7),
    transforms.RandomPerspective(distortion_scale=0.21, p=0.18),
    transforms.RandomPosterize(bits=4, p=0.79),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
