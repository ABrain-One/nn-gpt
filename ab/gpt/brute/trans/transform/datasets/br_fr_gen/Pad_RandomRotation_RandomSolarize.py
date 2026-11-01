import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(229, 188, 145), padding_mode='edge'),
    transforms.RandomRotation(degrees=11),
    transforms.RandomSolarize(threshold=74, p=0.54),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
