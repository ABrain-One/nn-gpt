import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.54, 0.8), ratio=(1.17, 2.84)),
    transforms.RandomPerspective(distortion_scale=0.13, p=0.76),
    transforms.RandomSolarize(threshold=83, p=0.38),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
