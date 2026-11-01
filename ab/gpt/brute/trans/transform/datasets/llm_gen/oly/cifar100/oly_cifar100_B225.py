import torch<br>
import torchvision.transforms as transforms<br>
<br>
def transform(norm):<br>
&nbsp;&nbsp;return transforms.Compose([<br>
&nbsp;&nbsp;&nbsp;transforms.RandomResizedCrop(size=32, scale=(0.6, 0.8), ratio=(1.19, 1.59)),<br>
&nbsp;&nbsp;&nbsp;transforms.Pad(padding=3, fill=(221, 195, 64), padding_mode='reflect'),<br>
&nbsp;&nbsp;&nbsp;transforms.RandomAdjustSharpness(sharpness_factor=1.71, p=0.56),<br>
&nbsp;&nbsp;&nbsp;transforms.Resize((224, 224)),<br>
&nbsp;&nbsp;&nbsp;transforms.ToTensor(),<br>
&nbsp;&nbsp;&nbsp;transforms.Normalize(*norm)<br>
&nbsp;&nbsp;])<br>