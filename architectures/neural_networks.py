import torch.nn as nn
import torch
import schemes
from architectures import convNd, resnet, r3plus1d
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


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, input):
        if self.dim:
            return torch.squeeze(input, self.dim)
        else:
            return torch.squeeze(input)


class R3plus1d(nn.Module):
    def __init__(self, config: schemes.R3plus1dRepresentationBaseInput):
        super(R3plus1d, self).__init__()
        self.params = config

        self.stem = r3plus1d.R3Plus1dStem(
            num_channels=self.params.num_channels,
        )
        self.layer = r3plus1d.Conv3Plus1D(
            in_planes=64,
            midplanes=128,
            out_planes=256,
            stride=1,
            padding=0,
        )
        self.dim_reduction = nn.Sequential(
            convNd.Conv4d(
                256,
                256,
                kernel_size=(1, 1, 1, 2),
                padding=(0, 0, 0, 0),
                stride=(1, 1, 1, 1),
                bias=False,
            ),
            convNd.BatchNorm4d(256),
            nn.ReLU(inplace=True),
            Squeeze(5),
        )
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            Squeeze(),
        )
        self.backbone = nn.Sequential(
            self.stem,
            self.layer,
            self.dim_reduction,
            self.avgpool,
        )

    def forward(self, x):
        # Conv4d - batch, days, channels, strikes, exps, types
        # x = self.stem(x)
        # x = self.layer(x)
        # x = self.dim_reduction(x)
        # x - self.avgpool(x)

        x = self.backbone(x)
        return x


class R3plus1dContrastiveLearner(R3plus1d):
    def __init__(self, config: schemes.R3plus1dRepresentationBaseInput):
        super().__init__(config)
        # self.contrastive_dim = config.contrastive_dim

        self.representation_layer = nn.Linear(
            in_features=256,  # NOTE match backbone?
            out_features=self.params.representation_dim,
        )
        # NOTE SimCLR specifically says nonlinearity helps aka 2 linlayers ^^^
        self.model = nn.Sequential(
            nn.Linear(
                in_features=self.params.representation_dim,
                out_features=int(self.params.representation_dim / 2),
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=int(self.params.representation_dim / 2),
                out_features=self.params.contrastive_dim,
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
        return (
            F.normalize(representation, dim=-1),
            F.normalize(contrastive_output, dim=-1),
        )


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

    def testR3plus1dContrastiveLearner():
        d_input = schemes.R3plus1dRepresentationBaseInput(
            num_channels=15,
            dropout=0.0,
            representation_dim=1024,
            contrastive_dim=512,
            output_channels=4,
        )
        contrastive_learner = R3plus1dContrastiveLearner(d_input)
        input = torch.randn(2, 15, 30, 180, 20, 2)
        c_output = contrastive_learner(input)
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
        # working
        r3plus1d_model = r3plus1d.VideoResNet(
            block=r3plus1d.BasicBlock,
            conv_makers=[
                r3plus1d.Conv3Plus1D,
                r3plus1d.Conv3Plus1D,
                resnet.Conv2Plus1D,
                resnet.Conv2Plus1D,
            ],
            layers=[2, 2, 2, 2],
            backbone=r3plus1d.R3Plus1dStem(num_channels=15),
        )
        # samples, channels, days, strikes, exps, types
        i = torch.randn(2, 15, 30, 180, 20, 2)
        o = r3plus1d_model(i)
        o

    def testR3plus1dDummy():
        # working
        r3plus1d = R3plus1d()  # num_channels=15)
        # samples, channels, days, strikes, exps, types
        i = torch.randn(2, 15, 30, 180, 20, 2)
        o = r3plus1d(i)
        o

    # testDiscriminatorAndContrastiveLearner()
    testR3plus1dContrastiveLearner()
    # testR2plus1d()
    # testR3plus1d()
    # testR3plus1dDummy()