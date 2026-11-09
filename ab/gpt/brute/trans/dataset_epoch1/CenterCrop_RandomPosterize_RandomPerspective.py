import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=25),
    transforms.RandomPosterize(bits=5, p=0.81),
    transforms.RandomPerspective(distortion_scale=0.18, p=0.43),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
