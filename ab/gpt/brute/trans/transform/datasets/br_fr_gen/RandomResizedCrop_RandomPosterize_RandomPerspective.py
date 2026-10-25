import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.53, 0.88), ratio=(1.1, 1.88)),
    transforms.RandomPosterize(bits=4, p=0.38),
    transforms.RandomPerspective(distortion_scale=0.26, p=0.77),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
