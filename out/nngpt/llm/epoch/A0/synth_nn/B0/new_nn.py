class Net(nn.Module):

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        layers = []
        in_channels = in_shape[1]
        layers += [nn.Conv2d(in_channels, 64, kernel_size=7, stride=3, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)]
        in_channels = 64
        layers += [nn.Conv2d(in_channels, 192, kernel_size=3, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)]
        in_channels = 192
        layers += [nn.Conv2d(in_channels, 440, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        in_channels = 440
        layers += [nn.Conv2d(in_channels, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        in_channels = 256
        layers += [nn.Conv2d(in_channels, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)]
        in_channels = 256
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        dropout_p = prm['dropout']
        classifier_input_features = in_channels * 6 * 6
        self.classifier = nn.Sequential(nn.Dropout(p=dropout_p), nn.Linear(classifier_input_features, 3072), nn.ReLU(inplace=True), nn.Dropout(p=dropout_p), nn.Linear(3072, 4096), nn.ReLU(inplace=True), nn.Linear(4096, out_shape[0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x