import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=0, fill=(40, 134, 218), padding_mode='reflect'),
    transforms.RandomAffine(degrees=17, translate=(0.04, 0.02), scale=(0.99, 1.99), shear=(3.94, 8.83)),
    transforms.RandomPerspective(distortion_scale=0.21, p=0.65),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
