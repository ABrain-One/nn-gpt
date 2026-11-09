import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(135, 28, 160), padding_mode='constant'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.51, p=0.37),
    transforms.RandomSolarize(threshold=102, p=0.81),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
