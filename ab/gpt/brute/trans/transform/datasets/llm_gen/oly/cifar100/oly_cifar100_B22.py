import torch<br>
import torchvision.transforms as transforms<br>
<br>
def transform(norm):<br>
&nbsp; &nbsp; return transforms.Compose([<br>
&nbsp; &nbsp; &nbsp; transforms.RandomAutocontrast(p=0.87),<br>
&nbsp; &nbsp; &nbsp; transforms.RandomHorizontalFlip(p=0.37),<br>
&nbsp; &nbsp; &nbsp; transforms.Resize((64,64)),<br>
&nbsp; &nbsp; &nbsp; transforms.ToTensor(),<br>
&nbsp; &nbsp; &nbsp; transforms.Normalize(*norm)<br>
&nbsp; &nbsp; ])