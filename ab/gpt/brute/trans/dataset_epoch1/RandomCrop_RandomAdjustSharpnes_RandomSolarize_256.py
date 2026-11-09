import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomCrop(size=26),
    transforms.RandomAdjustSharpness(sharpness_factor=0.72, p=0.29),
    transforms.RandomSolarize(threshold=206, p=0.5),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
