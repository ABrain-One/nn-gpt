import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(21, 164, 177), padding_mode='symmetric'),
    transforms.RandomCrop(size=29),
    transforms.RandomPerspective(distortion_scale=0.18, p=0.42),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
