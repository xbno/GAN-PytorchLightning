import torch.nn as nn
import torch
import schemes
from architectures import convNd, resnet
import torch.nn.functional as F

# ----------------------------------------------- #
# ---             ResNet3+1d                  --- #
# question
# where to reduce calls and puts into a single dim?
#
# have r(2+1)d for both calls and puts seperately
# until the end of the blocks?
#
# after the "stem"
# or
#
# ----------------------------------------------- #


class R3Plus1dStem(nn.Sequential):
    """R(3+1)D stem is different than the default one as it uses separated 4D convolution"""

    def __init__(self, num_channels):
        super(R3Plus1dStem, self).__init__(
            convNd.Conv4d(
                num_channels,
                45,
                kernel_size=(1, 7, 5, 1),  # (1, 14, 5, 1),  # (1,7,7,1)
                stride=(1, 2, 1, 1),  # (1, 2, 2, 1),
                padding=(0, 3, 2, 0),
                bias=False,
            ),
            convNd.BatchNorm4d(45),
            nn.ReLU(inplace=True),
            convNd.Conv4d(
                45,
                64,
                kernel_size=(3, 1, 1, 1),
                stride=(1, 1, 1, 1),
                padding=(1, 0, 0, 0),
                bias=False,
            ),
            convNd.BatchNorm4d(64),
            nn.ReLU(inplace=True),
        )


class Conv3Plus1D(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):
        super(Conv3Plus1D, self).__init__(
            convNd.Conv4d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3, 1),
                stride=(1, stride, stride, 1),
                padding=(0, padding, padding, 0),
                bias=False,
            ),
            convNd.BatchNorm4d(midplanes),
            nn.ReLU(inplace=True),
            convNd.Conv4d(
                midplanes,
                out_planes,
                kernel_size=(3, 1, 1, 1),
                stride=(stride, 1, 1, 1),
                padding=(padding, 0, 0, 0),
                bias=False,
            ),
        )

    @staticmethod
    def dims():
        return 4

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride, stride


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        conv_builder,
        stride=1,
        downsample=None,
    ):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            convNd.BatchNorm4d(planes)
            if conv_builder.dims() == 4
            else nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            convNd.BatchNorm4d(planes)
            if conv_builder.dims() == 4
            else nn.BatchNorm3d(planes),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# class Squeeze(nn.Module):
#     def __init__(self, dim=None):
#         super(Squeeze, self).__init__()
#         self.dim = dim

#     def forward(self, input):
#         if self.dim:
#             return torch.squeeze(input, self.dim)
#         else:
#             return torch.squeeze(input)


# class R3plus1d(nn.Module):
#     def __init__(self):
#         super(R3plus1d, self).__init__()

#         self.inplanes = 64
#         self.stem = R3Plus1dStem(
#             num_channels=15,
#         )
#         self.layer = Conv3Plus1D(
#             in_planes=64,
#             midplanes=128,
#             out_planes=256,
#             stride=1,
#             padding=0,
#         )
#         self.dim_reduction = nn.Sequential(
#             convNd.Conv4d(
#                 256,
#                 256,
#                 # maybe (1, 14, 3, 1) since ~300 strikes only ~20 exps
#                 kernel_size=(1, 1, 1, 2),
#                 # maybe (0, 0, 0, 0) because theres literally nothing existing beyond the edge
#                 padding=(0, 0, 0, 0),
#                 stride=(1, 1, 1, 1),
#                 bias=False,
#             ),
#             convNd.BatchNorm4d(256),
#             nn.ReLU(inplace=True),
#             Squeeze(5),
#         )
#         self.avgpool = nn.Sequential(
#             nn.AdaptiveAvgPool3d((1, 1, 1)),
#             Squeeze(),
#         )
#         self.backbone = nn.Sequential(
#             self.stem,
#             self.layer,
#             self.dim_reduction,
#             self.avgpool,
#         )

#     def forward(self, x):
#         # Conv4d - batch, days, channels, strikes, exps, types
#         # x = self.stem(x)
#         # x = self.layer(x)
#         # x = self.dim_reduction(x)
#         # x - self.avgpool(x)

#         x = self.backbone(x)
#         return x


class VideoResNet(nn.Module):
    def __init__(
        self,
        block,
        conv_makers,
        layers,
        backbone,
        num_classes=400,
        zero_init_residual=False,
    ):
        """Generic resnet video generator.
        Args:
            block (nn.Module): resnet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            stem (nn.Module, optional): Resnet stem, if None, defaults to conv-bn-relu. Defaults to None.
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = 64

        self.backbone = backbone

        self.layer1 = self._make_layer(
            block, conv_makers[0], 64, layers[0], stride=1, padding=0
        )
        self.layer2 = self._make_layer(
            block, conv_makers[1], 128, layers[1], stride=2, padding=0
        )
        self.dim_reduction = nn.Sequential(
            convNd.Conv4d(
                128,
                128,
                # maybe (1, 14, 3, 1) since ~300 strikes only ~20 exps
                kernel_size=(1, 1, 1, 2),
                # maybe (0, 0, 0, 0) because theres literally nothing existing beyond the edge
                padding=(0, 0, 0, 0),
                stride=(1, 1, 1, 1),
                bias=False,
            ),
            convNd.BatchNorm4d(128),
            nn.ReLU(inplace=True),
        )
        self.layer3 = self._make_layer(
            block, conv_makers[2], 256, layers[2], stride=2, padding=0
        )
        self.layer4 = self._make_layer(
            block, conv_makers[3], 512, layers[3], stride=2, padding=0
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        # Conv4d - batch, days, channels, strikes, exps, types
        x = self.backbone(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.squeeze(self.dim_reduction(x), 5)

        # Conv3d - batch, days, channels, strikes, exps
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1, padding=1):
        if conv_builder.dims() == 4:
            # if isinstance(conv_builder, convNd.Conv4d):
            downsample = None

            if stride != 1 or self.inplanes != planes * block.expansion:
                ds_stride = conv_builder.get_downsample_stride(stride)
                downsample = nn.Sequential(
                    convNd.Conv4d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=ds_stride,
                        padding=padding,
                        bias=False,
                    ),
                    convNd.BatchNorm4d(planes * block.expansion),
                )

        elif conv_builder.dims() == 3:
            # elif isinstance(conv_builder, nn.Conv3d):
            downsample = None

            if stride != 1 or self.inplanes != planes * block.expansion:
                ds_stride = conv_builder.get_downsample_stride(stride)
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=ds_stride,
                        padding=padding,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            # r2plus1
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # r3plus1
            elif isinstance(m, convNd.Conv4d):
                for conv_layer in m.conv_layers:
                    nn.init.kaiming_normal_(
                        conv_layer.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if conv_layer.bias is not None:
                        nn.init.constant_(conv_layer.bias, 0)
            elif isinstance(m, convNd.BatchNorm4d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # std
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":

    def testR2plus1d():
        r2plus1d = VideoResNet(
            block=resnet.BasicBlock,
            conv_makers=[resnet.Conv2Plus1D] * 4,
            layers=[2, 2, 2, 2],
            backbone=resnet.R2Plus1dStem(num_channels=15),
        )
        i = torch.randn(2, 15, 180, 20, 2)  # samples, channels, strikes, exps, types
        o = r2plus1d(i)
        o

    def testR3plus1d():
        # working
        r3plus1d = VideoResNet(
            block=BasicBlock,
            conv_makers=[
                Conv3Plus1D,
                Conv3Plus1D,
                resnet.Conv2Plus1D,
                resnet.Conv2Plus1D,
            ],
            layers=[2, 2, 2, 2],
            backbone=R3Plus1dStem(num_channels=15),
        )
        # r3plus1d = R3plus1d()
        # samples, channels, days, strikes, exps, types
        i = torch.randn(2, 15, 30, 180, 20, 2)
        o = r3plus1d(i)
        o

    # testDiscriminatorAndContrastiveLearner()
    # testR2p1DiscriminatorAndContrastiveLearner()
    # testR2plus1d()
    testR3plus1d()