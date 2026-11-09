import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomSolarize(threshold=47, p=0.78),
    transforms.ColorJitter(brightness=1.02, contrast=1.14, saturation=1.16, hue=0.06),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
