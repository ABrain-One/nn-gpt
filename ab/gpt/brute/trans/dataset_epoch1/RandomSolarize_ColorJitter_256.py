import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=210, p=0.82),
    transforms.ColorJitter(brightness=0.83, contrast=1.0, saturation=1.17, hue=0.09),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
