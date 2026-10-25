import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=1, fill=(5, 216, 154), padding_mode='symmetric'),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.72),
    transforms.RandomSolarize(threshold=20, p=0.51),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
