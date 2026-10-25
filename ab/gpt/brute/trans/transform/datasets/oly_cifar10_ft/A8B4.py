import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.RandomCrop(size=28),
        transforms.Pad(padding=4, fill=(134, 78, 215), padding_mode='symmetric'),
        transforms.RandomPerspective(distortion_scale=1.11, p=0.85),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
])