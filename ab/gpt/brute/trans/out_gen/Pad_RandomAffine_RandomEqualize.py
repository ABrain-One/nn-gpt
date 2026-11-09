import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
    transforms.Pad(padding=4, fill=(237, 244, 221), padding_mode='symmetric'),
    transforms.RandomAffine(degrees=16, translate=(0.07, 0.05), scale=(1.06, 1.47), shear=(1.82, 9.4)),
    transforms.RandomEqualize(p=0.19),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(*norm)
])
