import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=28),
    transforms.RandomSolarize(threshold=14, p=0.2),
    transforms.RandomEqualize(p=0.48),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
