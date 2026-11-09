import torch<br>
import torchvision.transforms as transforms<br>
<br>
def transform(norm):<br>
&nbsp;&nbsp;return transforms.Compose([<br>
&nbsp;&nbsp;&nbsp;transforms.Pad(padding=4, fill=(15, 198, 213), padding_mode='reflect'),<br>
&nbsp;&nbsp;&nbsp;transforms.RandomEqualize(p=0.16),<br>
&nbsp;&nbsp;&nbsp;transforms.RandomPosterize(bits=6, p=0.57),<br>
&nbsp;&nbsp;&nbsp;transforms.Resize((64, 64)),<br>
&nbsp;&nbsp;&nbsp;transforms.ToTensor(),<br>
&nbsp;&nbsp;&nbsp;transforms.Normalize(*norm)<br>
&nbsp;&nbsp;])<br>