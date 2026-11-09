import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.RandomCrop(size=25),
    transforms.RandomPerspective(distortion_scale=0.14, p=0.2),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
