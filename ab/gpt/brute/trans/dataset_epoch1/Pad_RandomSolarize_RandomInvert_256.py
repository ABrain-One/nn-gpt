import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=5, fill=(103, 181, 121), padding_mode='edge'),
    transforms.RandomSolarize(threshold=195, p=0.89),
    transforms.RandomInvert(p=0.52),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
