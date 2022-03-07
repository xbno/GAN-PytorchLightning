import torch.nn as nn
import torch
import schemes
from architectures import convNd, resnet
import torch.nn.functional as F

# --------------------------------------------------------------- #
# --- To keep track of the dimensions of convolutional layers --- #
# --------------------------------------------------------------- #
class PrintShape(nn.Module):
    def __init__(self, message: str):
        super(PrintShape, self).__init__()
        self.message = message

    def forward(self, feat: torch.Tensor):
        print(self.message, feat.shape)
        return feat


# ----------------------------------------------- #
# --- Convolutional Generator & Distriminator --- #
# ----------------------------------------------- #
class ConvolutionalGenerator(nn.Module):
    def __init__(self, config: schemes.GeneratorInput):
        super(ConvolutionalGenerator, self).__init__()
        self.params = config

        self.model = nn.Sequential(
            nn.Linear(self.params.latent_dim, 4096),
            nn.Unflatten(1, (256, 4, 4)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout(p=self.params.dropout),
            nn.ConvTranspose3d(256, 128, 3, stride=2, padding=1),  # 128, 7, 7
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Dropout(p=self.params.dropout),
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=0),  # 64, 16, 16
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout(p=self.params.dropout),
            nn.ConvTranspose3d(
                64, self.params.output_channels, 4, stride=2, padding=1
            ),  # 3, 32, 32
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)


class ConvolutionalBase(nn.Module):
    def __init__(self, config: schemes.RepresentationBaseInput):
        super(ConvolutionalBase, self).__init__()
        self.params = config

        self.backbone = nn.Sequential(
            nn.Conv3d(self.params.output_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.params.dropout),
            nn.Conv3d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.params.dropout),
            nn.Conv3d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(start_dim=1),
        )

    def forward(self, x):
        return self.backbone(x)


class Discriminator(ConvolutionalBase):
    def __init__(self, config: schemes.RepresentationBaseInput):
        super().__init__(config)

        self.model = nn.Sequential(
            nn.Linear(4 * 4 * 256, 1024), nn.ReLU(), nn.Linear(1024, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        mid_representation = self.backbone(x)
        prediction = self.model(mid_representation).squeeze()
        return prediction


class ContrastiveLearner(ConvolutionalBase):
    def __init__(self, config: schemes.RepresentationBaseInput):
        super().__init__(config)
        self.contrastive_dim = config.contrastive_dim

        self.representation_layer = nn.Linear(
            4 * 4 * 256, self.params.representation_dim
        )
        self.model = nn.Sequential(
            nn.Linear(
                self.params.representation_dim, int(self.params.representation_dim / 2)
            ),
            nn.ReLU(),
            nn.Linear(
                int(self.params.representation_dim / 2), self.params.contrastive_dim
            ),
        )

    def get_representations(self, x: torch.Tensor):
        base_representation = self.backbone(x)
        representation = self.representation_layer(base_representation)
        return representation

    def forward(self, x: torch.Tensor):
        base_representation = self.backbone(x)
        representation = self.representation_layer(base_representation)
        contrastive_output = self.model(representation)
        return F.normalize(representation, dim=-1), F.normalize(
            contrastive_output, dim=-1
        )


# ----------------------------------------------- #
# ---             ResNet2+1d                  --- #
# ---  https://arxiv.org/pdf/1711.11248.pdf   --- #
# ----------------------------------------------- #
# https://github.com/pytorch/vision/blob/5a315453da5089d66de94604ea49334a66552524/torchvision/models/video/resnet.py#L275


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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):

        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            convNd.Conv4d(inplanes, planes, kernel_size=1, bias=False),
            convNd.BatchNorm4d(planes),
            nn.ReLU(inplace=True),
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            convNd.BatchNorm4d(planes),
            nn.ReLU(inplace=True),
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            convNd.Conv4d(planes, planes * self.expansion, kernel_size=1, bias=False),
            convNd.BatchNorm4d(planes * self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


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


class R2p1Discriminator(VideoResNet):
    def __init__(self, config: schemes.R2p1RepresentationBaseInput):
        super().__init__(config)

        self.model = nn.Sequential(
            nn.Linear(4 * 4 * 256, 1024), nn.ReLU(), nn.Linear(1024, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor):
        mid_representation = self.backbone(x)
        prediction = self.model(mid_representation).squeeze()
        return prediction


if __name__ == "__main__":

    def testGenerator():
        gen_input = schemes.GeneratorInput(latent_dim=50, dropout=0.0)
        gen = ConvolutionalGenerator(gen_input)
        input = torch.randn(2, 50)
        output = gen(input)
        print(output.shape)

    def testDiscriminatorAndContrastiveLearner():
        d_input = schemes.RepresentationBaseInput(
            dropout=0.0,
            representation_dim=1024,
            contrastive_dim=512,
            output_channels=4,
        )
        discriminator = Discriminator(d_input)
        contrastive_learner = ContrastiveLearner(d_input)
        input = torch.randn(2, 4, 180, 20, 2)
        d_output = discriminator(input)
        c_output = contrastive_learner(input)
        print(d_output.shape)
        print(c_output[0].shape, c_output[1].shape)
        print("Test")

    def testR2p1DiscriminatorAndContrastiveLearner():
        d_input = schemes.R2p1RepresentationBaseInput(
            dropout=0.0,
            representation_dim=1024,
            contrastive_dim=512,
            output_channels=4,
            block=BasicBlock,
            conv_makers=[Conv2Plus1D] * 4,
            layers=[2, 2, 2, 2],
            backbone=R2Plus1dStem(num_channels=15),
        )
        discriminator = R2p1Discriminator(d_input)
        contrastive_learner = ContrastiveLearner(d_input)
        input = torch.randn(2, 4, 180, 20, 2)
        d_output = discriminator(input)
        c_output = contrastive_learner(input)
        print(d_output.shape)
        print(c_output[0].shape, c_output[1].shape)
        print("Test")

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
        # samples, channels, days, strikes, exps, types
        i = torch.randn(2, 15, 30, 180, 20, 2)
        o = r3plus1d(i)
        o

    # testDiscriminatorAndContrastiveLearner()
    # testR2p1DiscriminatorAndContrastiveLearner()
    # testR2plus1d()
    testR3plus1d()