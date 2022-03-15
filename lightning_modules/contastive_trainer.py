import torch
from schemes import (
    # GANBaseInput,
    GANTrainInput,
    # GeneratorInput,
    R3plus1dRepresentationBaseInput,
)
from architectures.neural_networks import (
    ConvolutionalGenerator,
    Discriminator,
    R3plus1dContrastiveLearner,
)
from pytorch_lightning import LightningModule
from collections import OrderedDict
import torchvision
import torch.optim as optim
from typing import Optional
import numpy as np
import pl_bolts


class ContrastiveR3plus1d(LightningModule):
    def __init__(
        self,
        model_config: R3plus1dRepresentationBaseInput,
        train_config: GANTrainInput,
        experiment_id: str,
    ):
        super(ContrastiveR3plus1d, self).__init__()
        self.save_hyperparameters()
        self.experiment_id = experiment_id
        self.model_params = model_config
        # self.generator = self.init_generator()
        # self.discriminator = self.init_discriminator()
        self.represtentation_learner = self.init_representation_learner()
        self.train_params = train_config
        self.fixed_validation_z = self.sample_latent_noise(8)

    def init_representation_learner(self) -> R3plus1dContrastiveLearner:
        representation_learner_config = R3plus1dRepresentationBaseInput(
            representation_dim=self.model_params.representation_dim,
            dropout=self.model_params.discriminator_dropout,
            output_channels=self.model_params.output_channels,
            contrastive_dim=self.model_params.contrastive_dim,
        )
        return R3plus1dContrastiveLearner(representation_learner_config)

    # ---
    # --- Utils functions --- #
    def dinput_noise(self, tensor):
        """Adds small Gaussian noise to the tensor."""
        dinput_noise = torch.zeros(tensor.size()).type_as(tensor)
        return tensor + dinput_noise

    def sample_latent_noise(self, batch_size):
        """Generates gaussian noise."""
        z_noise = torch.empty(
            (batch_size, self.generator.params.latent_dim), requires_grad=False
        ).normal_()  # b, z_dim
        return z_noise

    # ---
    # --- Pytorch ligthning --- #
    def forward(
        self,
        n_samples: Optional[int] = 8,
        return_imgs: bool = True,
        imgs: Optional[torch.Tensor] = None,
        return_representations: bool = False,
    ):
        if return_imgs:
            assert n_samples is not None
            z = self.sample_latent_noise(n_samples)
            generated_imgs = self.generator(z)
            if not return_representations:
                return generated_imgs
        if return_representations:
            assert imgs is not None
            representations = self.represtentation_learner.get_representations(imgs)
            if not return_imgs:
                return representations
        return generated_imgs, representations

    def configure_optimizers(self):
        optimiser_R = optim.Adam(
            self.represtentation_learner.parameters(),
            lr=self.train_params.representation_lr,
            betas=(
                self.train_params.representation_beta1,
                self.train_params.representation_beta2,
            ),
        )
        optimiser_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.train_params.discriminator_lr,
            betas=(
                self.train_params.discriminator_beta1,
                self.train_params.discriminator_beta2,
            ),
        )
        optimiser_G = optim.Adam(
            self.generator.parameters(),
            lr=self.train_params.generator_lr,
            betas=(
                self.train_params.generator_beta1,
                self.train_params.generator_beta2,
            ),
        )
        return [optimiser_R, optimiser_D, optimiser_G], []

    def gan_loss(self, y_pred, y_true):
        return torch.nn.BCELoss()(y_pred, y_true)

    def contrastive_loss(self):
        pl_bolts.losses.self_supervised_learning.FeatureMapContrastiveTask()

    def contrastive_loss(
        self, out_1: torch.Tensor, out_2: torch.Tensor, batch_size: int
    ):
        out = torch.cat([out_1, out_2], dim=0)  # [2*B, D]
        sim_matrix = torch.exp(
            torch.mm(out, out.t().contiguous())
            / self.train_params.contrastive_temperature
        )  # [2*B, 2*B]
        mask = (
            torch.ones_like(sim_matrix).type_as(out_1)
            - torch.eye(2 * batch_size).type_as(out_1)
        ).bool()  # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute NTX loss
        pos_sim = torch.exp(
            torch.sum(out_1 * out_2, dim=-1) / self.train_params.contrastive_temperature
        )  # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (-torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        return loss

    def compute_feature_distance(
        self, features1: torch.Tensor, features2: torch.Tensor
    ):
        current_bs = features1.shape[0]
        features1_idxs = np.tile(
            np.arange(current_bs), current_bs
        )  # [0, ..., bs, 0, ..., bs, ..]
        features2_idxs = np.tile(np.arange(current_bs), (current_bs, 1)).T.reshape(
            -1
        )  # [0, ..., 0, 1, ..., 1, ...]
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        distances = 1 - cos(features1[features1_idxs], features2[features2_idxs])
        distances = distances.reshape(current_bs, current_bs)
        positive_distances = torch.diagonal(
            distances
        ).clone()  # distances of positive pairs, [bs]
        negative_distances = torch.sum(distances.fill_diagonal_(0), dim=1) / (
            current_bs - 1
        )  # [bs]
        return torch.mean(positive_distances), torch.mean(negative_distances)

    def training_step(self, batch, batch_idx, optimizer_idx):

        # --- Train the Representation Learner --- #
        if optimizer_idx == 0:
            imgs_1, imgs_2, _ = batch["contrastive"]
            batch_size = imgs_1.shape[0]
            # TODO: contstruct negative pairs from generated images
            features_1, out_1 = self.represtentation_learner(imgs_1)
            features_2, out_2 = self.represtentation_learner(imgs_2)
            r_loss = self.contrastive_loss(out_1, out_2, batch_size)
            batch_pos_distances, batch_neg_distances = self.compute_feature_distance(
                features_1, features_2
            )
            tqdm_dict = {
                "r_loss": r_loss,
                "batch_pos_distances": batch_pos_distances,
                "batch_neg_distances": batch_neg_distances,
            }
            output = OrderedDict(
                {"loss": r_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

        else:
            # Ground truths
            real_imgs, _ = batch["GAN"]
            batch_size = real_imgs.shape[0]

            real_labels = torch.ones(batch_size).type_as(real_imgs)
            fake_labels = torch.zeros(batch_size).type_as(real_imgs)
            z_noise = self.sample_latent_noise(batch_size).type_as(real_imgs)

            # --- Train the Discriminator --- #
            if optimizer_idx == 1:

                # Loss for real images
                real_imgs = self.dinput_noise(real_imgs)
                real_pred = self.discriminator(real_imgs)
                assert torch.sum(torch.isnan(real_pred)) == 0, real_pred
                assert (real_pred >= 0.0).all(), real_pred
                assert (real_pred <= 1.0).all(), real_pred
                d_real_loss = self.gan_loss(real_pred, real_labels)
                D_img = real_pred.mean()

                # Loss for fake images
                fake_imgs = self.generator(z_noise)
                assert torch.sum(torch.isnan(fake_imgs)) == 0, fake_imgs
                fake_imgs_input = self.dinput_noise(fake_imgs.detach())
                fake_pred = self.discriminator(fake_imgs_input)
                assert (fake_pred >= 0.0).all(), fake_pred
                assert (fake_pred <= 1.0).all(), fake_pred
                d_fake_loss = self.gan_loss(fake_pred, fake_labels)
                D_G_z1 = fake_pred.mean()

                d_loss = (d_real_loss + d_fake_loss) / 2
                tqdm_dict = {"d_loss": d_loss, "D_img": D_img, "D_G_z1": D_G_z1}
                output = OrderedDict(
                    {"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
                )
                return output

            # --- Train the Generator --- #
            if optimizer_idx == 2:
                fake_imgs = self.generator(z_noise)
                fake_imgs_input = self.dinput_noise(fake_imgs)
                fake_pred = self.discriminator(fake_imgs_input)
                g_loss = self.gan_loss(fake_pred, real_labels)
                D_G_z2 = fake_pred.mean()

                # log sampled images
                sample_imgs = fake_imgs[:6]
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image(
                    "generated_images_after_update", grid, 0
                )

                tqdm_dict = {"g_loss": g_loss, "D_G_z2": D_G_z2}
                output = OrderedDict(
                    {"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
                )
                return output

    def training_epoch_end(self, outputs: list):
        r_outputs, d_outputs, g_outputs = outputs

        for log_key in r_outputs[0]["log"]:
            avg_batch_log = torch.stack(
                [batch_res["log"][log_key] for batch_res in r_outputs]
            ).mean()
            self.logger.experiment.add_scalar(
                f"{log_key}/Train", avg_batch_log, self.current_epoch
            )

        avg_losses = {}
        for log_key in d_outputs[0]["log"]:
            avg_batch_log = torch.stack(
                [batch_res["log"][log_key] for batch_res in d_outputs]
            ).mean()
            self.logger.experiment.add_scalar(
                f"{log_key}/Train", avg_batch_log, self.current_epoch
            )
            if log_key == "d_loss":
                avg_losses[log_key] = avg_batch_log
        for log_key in g_outputs[0]["log"]:
            avg_batch_log = torch.stack(
                [batch_res["log"][log_key] for batch_res in g_outputs]
            ).mean()
            self.logger.experiment.add_scalar(
                f"{log_key}/Train", avg_batch_log, self.current_epoch
            )
            if log_key == "g_loss":
                avg_losses[log_key] = avg_batch_log

        self.logger.experiment.add_scalars(
            "Losses/Train", avg_losses, self.current_epoch
        )
        self.log_generated_images("generated_images")

    # def on_epoch_end(self) -> None:
    #     # log sampled images
    #     self.log_generated_images('generated_images')

    def log_generated_images(self, name: str):
        z = self.fixed_validation_z.type_as(self.generator.model[0].weight)
        sample_imgs = self.generator(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(name, grid, self.current_epoch)


if __name__ == "__main__":
    from pl_bolts.losses.self_supervised_learning import (
        FeatureMapContrastiveTask,
        nt_xent_loss,
    )

    # actual use case is

    # idea is a different than the standard approach which is something like this:
    # https://github.com/khuongnd/advanced_ml_algorithms/blob/daec642df355b54572b742de25910420d812dcfd/02_self_supervised/amdim/amdim_module.py
    # training step
    # [img_1, img_2], _ = batch
    # # extract from diff resnet blocks
    # r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2 = forward(img_1, img_2)
    # # training step end, calc loss
    # loss, lgt_reg = FeatureMapContrastiveTask('01, 02, 11')((r1_x1, r5_x1, r7_x1), (r1_x2, r5_x2, r7_x2))
    # unsupervised_loss = loss.sum() + lgt_reg

    # instead i'll do less representation comparisons but include an extra anc -> pos by using symbol as the
    # "augmentation" instead of time. so there will be 3x anc:pos matches instead of 2
    # aug: same symbol diff time
    # anc1: timeframe_1_symbol_1
    # pos1: timeframe_2_symbol_1

    # aug: same symbol diff time
    # anc2: timeframe_2_symbol_2
    # pos2: timeframe_3_symbol_2

    # aug: diff symbols same time
    # pos1: timeframe_2_symbol_1
    # anc2: timeframe_2_symbol_2

    # # FeatureMapContrastiveTask('01, 02, 11')((representations_of_anchor), (representations_of_anchor_aug))
    # # training step
    # [
    #     timeframe_1_symbol_1,
    #     timeframe_2_symbol_1,
    #     timeframe_2_symbol_2,
    #     timeframe_3_symbol_2,
    # ] = batch
    # rep_t1_s1, rep_t2_s1, rep_t2_s2, rep_t3_s2 = forward(
    #     timeframe_1_symbol_1,
    #     timeframe_2_symbol_1,
    #     timeframe_2_symbol_2,
    #     timeframe_3_symbol_2,
    # )
    # FeatureMapContrastiveTask("00, 11, 10")(
    #     (rep_t1_s1, rep_t2_s2),
    #     (rep_t2_s1, rep_t3_s2),
    # )

    b = 24
    k = 5
    h = 1
    w = 1

    # first run x1 and x2 through encoder which is results in
    pass

    # then
    # r1_anchors = torch.rand(b, k, h, w)
    # r3_anchors = torch.rand(b, k, h * 2, w * 2)
    # r5_anchors = torch.rand(b, k, h * 4, w * 4)

    # r1_pos = torch.rand(b, k, h, w)
    # r3_pos = torch.rand(b, k, h * 2, w * 2)
    # r5_pos = torch.rand(b, k, h * 4, w * 4)

    # task = FeatureMapContrastiveTask("01, 02, 11")
    # losses, regularizer = task(
    #     (r1_anchors, r3_anchors, r5_anchors),
    #     (r1_pos, r3_pos, r5_pos),
    # )

    # don't understand the pair selection, so..
    r1_anchors = torch.rand(b, k, h, w)
    r1_pos = torch.rand(b, k, h, w)
    task = FeatureMapContrastiveTask("00")
    losses, regularizer = task(
        (r1_anchors,),
        (r1_pos,),
    )

    # probably just stick to simclr style single anchor/pos_aug with 1 feat vector per
    r1_anchor_flattened = torch.rand(b, k)
    r1_pos_flattened = torch.rand(b, k)
    loss = nt_xent_loss(r1_anchor_flattened, r1_pos_flattened, 0.5)
