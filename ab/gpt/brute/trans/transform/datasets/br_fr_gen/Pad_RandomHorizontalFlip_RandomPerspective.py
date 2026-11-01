import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(139, 30, 204), padding_mode='symmetric'),
    transforms.RandomHorizontalFlip(p=0.67),
    transforms.RandomPerspective(distortion_scale=0.21, p=0.8),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
