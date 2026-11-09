import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomPerspective(distortion_scale=0.16, p=0.39),
    transforms.RandomPosterize(bits=8, p=0.5),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
