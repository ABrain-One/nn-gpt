import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.84),
    transforms.RandomAdjustSharpness(sharpness_factor=1.65, p=0.9),
    transforms.RandomSolarize(threshold=216, p=0.31),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
