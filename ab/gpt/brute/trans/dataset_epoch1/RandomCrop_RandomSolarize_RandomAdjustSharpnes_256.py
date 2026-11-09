import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=30),
    transforms.RandomSolarize(threshold=37, p=0.45),
    transforms.RandomAdjustSharpness(sharpness_factor=0.54, p=0.28),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
