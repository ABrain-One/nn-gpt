<td>
<pre>
import torch
import torchvision.transforms as transforms

def transform(norm):
    return transforms.Compose([
        transforms.Pad(padding=0, fill=(25, 148, 180), padding_mode='reflect'),
        transforms.RandomVerticalFlip(p=0.69),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize(**norm)
    ])
</pre>
</td>