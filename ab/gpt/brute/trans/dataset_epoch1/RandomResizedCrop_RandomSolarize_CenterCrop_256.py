import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomResizedCrop(size=32, scale=(0.77, 0.82), ratio=(0.92, 2.8)),
    transforms.RandomSolarize(threshold=5, p=0.18),
    transforms.CenterCrop(size=29),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
