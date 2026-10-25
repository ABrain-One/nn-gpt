import torch<br/>
import torchvision.transforms as transforms<br/>
<br/>
def transform(norm):<br/>
&nbsp; &nbsp; return transforms.Compose([<br/>
&nbsp; &nbsp; &nbsp; transforms.Pad(padding=3, fill=(115, 217, 221), padding_mode='symmetric'),<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomInvert(p=0.31),<br/>
&nbsp; &nbsp; &nbsp; transforms.RandomHorizontalFlip(p=0.77),<br/>
&nbsp; &nbsp; &nbsp; transforms.Resize((64,64)),<br/>
&nbsp; &nbsp; &nbsp; transforms.ToTensor(),<br/>
&nbsp; &nbsp; &nbsp; transforms.Normalize(*norm)<br/>
&nbsp; &nbsp; ])