<pre>
import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.RandomCrop(27),
        transforms.RandomVerticalFlip(p=0.54),
        transforms.RandomAffine(21, (0.11, 0.16), (1.04, 1.33), (2.43, 6.72)),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(**norm)
    ])
</pre>