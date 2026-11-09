import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=2, fill=(252, 249, 223), padding_mode='reflect'),
    transforms.CenterCrop(size=27),
    transforms.RandomSolarize(threshold=19, p=0.22),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
