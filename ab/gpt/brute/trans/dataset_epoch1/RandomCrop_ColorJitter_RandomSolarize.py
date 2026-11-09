import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=25),
    transforms.ColorJitter(brightness=1.02, contrast=1.18, saturation=1.06, hue=0.06),
    transforms.RandomSolarize(threshold=221, p=0.54),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
