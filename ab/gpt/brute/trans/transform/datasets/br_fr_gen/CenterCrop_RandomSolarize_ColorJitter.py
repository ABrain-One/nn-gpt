import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.CenterCrop(size=31),
    transforms.RandomSolarize(threshold=134, p=0.13),
    transforms.ColorJitter(brightness=0.96, contrast=1.18, saturation=1.03, hue=0.01),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
