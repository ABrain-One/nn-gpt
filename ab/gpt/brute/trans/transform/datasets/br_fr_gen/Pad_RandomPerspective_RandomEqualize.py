import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(167, 252, 66), padding_mode='edge'),
    transforms.RandomPerspective(distortion_scale=0.18, p=0.2),
    transforms.RandomEqualize(p=0.49),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
