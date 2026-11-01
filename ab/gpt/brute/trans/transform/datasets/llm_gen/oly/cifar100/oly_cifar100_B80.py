import torch<br>
import torchvision.transforms as transforms<br>
<br>
def transform(norm):<br>
&nbsp; &nbsp; return transforms.Compose([<br>
&nbsp; &nbsp; &nbsp; transforms.RandomGrayscale(p=0.33),<br>
&nbsp; &nbsp; &nbsp; transforms.RandomSolarize(threshold=243, p=0.27),<br>
&nbsp; &nbsp; &nbsp; transforms.Resize(224),<br>
&nbsp; &nbsp; &nbsp; transforms.ToTensor(),<br>
&nbsp; &nbsp; &nbsp; transforms.Normalize(*norm)<br>
&nbsp; &nbsp; ])