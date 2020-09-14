import torch.nn as nn
from torchsummary import summary


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),      # (224 + 2 * 2 - 11) / 4 + 1 = 55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                      # (55 - 3) / 2 + 1 = 27

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),     # (27 + 2 * 2 - 5) / 1 + 1 = 27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                      # (27 - 3) / 2 + 1 = 13

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),    # (13 + 1 * 2 - 3) / 1 + 1 = 13
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),    # (13 + 1 * 2 - 3) / 1 + 1 = 13
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),    # (13 + 1 * 2 - 3) / 1 + 1 = 13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                      # (13 - 3) / 2 + 1 = 6
        )   # 6 * 6 * 256 = 9126

        # 保证不管在输入图在什么尺度下，都能保证生成的特征图是6*6的
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))                       # 输出特征图的大小已经是6*6的了，为什么在这里额外加一个平均池化


        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6*6*256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

def alexnet():
    model = AlexNet()
    return model

if __name__ == "__main__":
    model = alexnet()
    summary(model, (3, 224, 224))
    