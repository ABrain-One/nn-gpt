import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.ColorJitter(brightness=0.96, contrast=0.91, saturation=0.86, hue=0.02),
    transforms.RandomCrop(size=29),
    transforms.RandomSolarize(threshold=104, p=0.46),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
