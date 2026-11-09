import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(41, 242, 115), padding_mode='reflect'),
    transforms.RandomRotation(degrees=17),
    transforms.RandomPerspective(distortion_scale=0.25, p=0.51),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
