import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=26),
    transforms.RandomSolarize(threshold=221, p=0.65),
    transforms.ColorJitter(brightness=1.03, contrast=1.2, saturation=0.94, hue=0.08),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
