import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.RandomVerticalFlip(p=0.62),
    transforms.RandomPerspective(distortion_scale=0.28, p=0.89),
    transforms.Pad(padding=1, fill=(108, 134, 41), padding_mode='symmetric'),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
