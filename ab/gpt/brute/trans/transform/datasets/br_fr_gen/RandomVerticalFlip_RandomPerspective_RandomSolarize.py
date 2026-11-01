import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.45),
    transforms.RandomPerspective(distortion_scale=0.22, p=0.42),
    transforms.RandomSolarize(threshold=6, p=0.32),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
