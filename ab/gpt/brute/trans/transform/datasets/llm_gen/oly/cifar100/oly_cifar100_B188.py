<code>
        import&nbsp;torch<br/>
        import&nbsp;torchvision.transforms&nbsp;as&nbsp;transforms<br/>
        <br/>
        def&nbsp;transform(norm):<br/>
            &nbsp;&nbsp;return&nbsp;transforms.Compose([<br/>
            &nbsp;&nbsp;&nbsp;&nbsp;transforms.Pad(padding=4, fill=(72, 143, 255), padding_mode='constant'),<br/>
            &nbsp;&nbsp;&nbsp;&nbsp;transforms.RandomRotation(degrees=15, resample=Image.BICUBIC, expand=True),<br/>
            &nbsp;&nbsp;&nbsp;&nbsp;transforms.RandomPerspective(distortion_scale=0.3, p=0.5),<br/>
            &nbsp;&nbsp;&nbsp;&nbsp;transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),<br/>
            &nbsp;&nbsp;&nbsp;&nbsp;transforms.Resize((64, 64)),<br/>
            &nbsp;&nbsp;&nbsp;&nbsp;transforms.ToTensor(),<br/>
            &nbsp;&nbsp;&nbsp;&nbsp;transforms.Normalize(**norm)<br/>
            &nbsp;&nbsp;])
    </code>