import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(168, 167, 134), padding_mode='symmetric'),
    transforms.RandomPerspective(distortion_scale=0.21, p=0.79),
    transforms.RandomGrayscale(p=0.43),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
