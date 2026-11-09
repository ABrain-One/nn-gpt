import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=1.05, contrast=1.04, saturation=0.97, hue=0.09),
    transforms.CenterCrop(size=29),
    transforms.RandomSolarize(threshold=93, p=0.76),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
