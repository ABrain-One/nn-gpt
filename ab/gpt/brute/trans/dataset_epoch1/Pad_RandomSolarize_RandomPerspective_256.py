import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=3, fill=(210, 68, 250), padding_mode='symmetric'),
    transforms.RandomSolarize(threshold=221, p=0.17),
    transforms.RandomPerspective(distortion_scale=0.21, p=0.44),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
