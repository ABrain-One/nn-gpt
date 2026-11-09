import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=25),
    transforms.RandomHorizontalFlip(p=0.68),
    transforms.RandomPerspective(distortion_scale=0.13, p=0.19),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
