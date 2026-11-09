import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=25),
    transforms.RandomRotation(degrees=9),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
