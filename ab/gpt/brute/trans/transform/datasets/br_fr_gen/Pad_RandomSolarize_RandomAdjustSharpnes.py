import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(93, 239, 17), padding_mode='constant'),
    transforms.RandomSolarize(threshold=163, p=0.25),
    transforms.RandomAdjustSharpness(sharpness_factor=1.68, p=0.86),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
