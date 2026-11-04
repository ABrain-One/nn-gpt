import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(152, 112, 137), padding_mode='symmetric'),
    transforms.RandomPerspective(distortion_scale=0.25, p=0.37),
    transforms.RandomVerticalFlip(p=0.27),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
