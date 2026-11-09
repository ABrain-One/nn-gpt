import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=27),
    transforms.RandomResizedCrop(size=32, scale=(0.65, 0.83), ratio=(0.94, 2.5)),
    transforms.RandomRotation(degrees=17),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
