<td>
import torch<br>
import torchvision.transforms as transforms<br>
<br>
def transform(norm):<br>
    return transforms.Compose([<br>
        transforms.CenterCrop(size=30),<br>
        transforms.RandomEqualize(p=0.73),<br>
        transforms.Resize((64,64)),<br>
        transforms.ToTensor(),<br>
        transforms.Normalize(*norm)<br>
    ])
</td>