import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(55, 69, 103), padding_mode='symmetric'),
    transforms.RandomPosterize(bits=4, p=0.61),
    transforms.RandomAdjustSharpness(sharpness_factor=1.72, p=0.29),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])