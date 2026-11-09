import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.26),
    transforms.RandomSolarize(threshold=138, p=0.64),
    transforms.RandomPerspective(distortion_scale=0.26, p=0.67),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
