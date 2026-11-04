<td>&lt;?xml version="1.0" encoding="UTF-8"?&gt;
&lt;tr&gt;
&lt;code&gt;
&lt;pre&gt;
import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),
        transforms.RandomRotation(degrees=(-5, 5)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(*norm)
    ])
&lt;/pre&gt;
&lt;/code&gt;
&lt;/tr&gt;