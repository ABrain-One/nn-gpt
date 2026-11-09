import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(145, 147, 25), padding_mode='symmetric'),
    transforms.RandomPosterize(bits=8, p=0.86),
    transforms.RandomSolarize(threshold=181, p=0.49),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
