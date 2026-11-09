import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(70, 155, 34), padding_mode='symmetric'),
    transforms.RandomPerspective(distortion_scale=0.23, p=0.88),
    transforms.RandomInvert(p=0.16),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
