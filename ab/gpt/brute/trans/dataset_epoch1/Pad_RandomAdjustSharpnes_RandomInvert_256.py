import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(79, 192, 34), padding_mode='reflect'),
    transforms.RandomAdjustSharpness(sharpness_factor=0.57, p=0.69),
    transforms.RandomInvert(p=0.57),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
