from schemes import DataInput
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

import torch
import sys

sys.path.append("/Users/xbno/options_trading")
import glob
import itertools
import random
import numpy as np
from lambdas import yf_utils, json_logger

from pl_cnn import data
import argparse
import logging
import torch
from torch.utils.data import Dataset, DataLoader, sampler

json_logger.set_aws_logging("info")


def pad_to_dim(v, dim):
    n, c, d_diff, h_diff, w = dim - torch.Tensor([*v.shape])
    w_left, w_right = 0, 0
    h_top, h_bottom = 0, h_diff
    d_front, d_back = 0, d_diff
    pad = (
        int(w_left),
        int(w_right),
        int(h_top),
        int(h_bottom),
        int(d_front),
        int(d_back),
    )
    return torch.nn.functional.pad(v, pad=pad)


class PadCollate:
    def __init__(self, num_frames):
        self.num_frames = num_frames

    def pad_collate(self, batch):
        max_slice_in_batch_shape = torch.cat(
            [
                torch.stack([torch.Tensor([*symbol.shape]) for symbol in sample])
                for sample in batch
            ]
        ).max(0)[0]

        # pad batch to largest timeseries
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                batch[i][j] = pad_to_dim(
                    batch[i][j],
                    dim=max_slice_in_batch_shape,
                )

        # split each symbol timeseries into the [timeseries_a : time_delta : timeseries_b]
        split_batch = []
        for i in range(len(batch)):
            split_sample = []
            for j in range(len(batch[i])):
                # split_sample.extend([batch[i][j][:self.num_frames])
                split_sample.append(batch[i][j][self.num_frames :])
                split_sample.append(batch[i][j][: self.num_frames])
            split_batch.append(split_sample)

        # batch
        return split_batch

    def __call__(self, batch):
        return {"train_data": self.pad_collate(batch)}


# def spatiotemporal_symbol_set(
#     dataset_path,
#     symbols,
#     dates,
#     num_frames=2,
#     stride=1,
#     max_delta=3,
#     num_sets=10,
# ):

#     full_sets = []
#     deltas = np.random.choice(
#         np.arange(max_delta),
#         p=np.arange(max_delta, 0, -1) / sum(np.arange(max_delta, 0, -1)),
#         size=num_sets,
#     )
#     symbol_pairs = list(itertools.combinations(symbols, 2))
#     random.shuffle(symbol_pairs)
#     for (symbol1, symbol2), delta in zip(symbol_pairs[:num_sets], deltas):
#         if len(dates) <= num_frames * 3 + delta * 2:
#             logging.warning(
#                 f"num_frames({num_frames}) and delta({delta}): {num_frames * 3 + delta * 2}d sample >> {len(dates)}d dataset"
#                 # + f"{num_frames}d + {delta}d + {num_frames}d + {delta}d + {num_frames}d"
#             )
#         date_set = [
#             (
#                 dates[x : x + num_frames],
#                 dates[x + num_frames + delta : x + num_frames * 2 + delta],
#                 dates[x + num_frames * 2 + delta : x + num_frames * 3 + delta],
#             )
#             for x in range(0, len(dates), stride)
#             if len(dates[x + num_frames * 2 + delta : x + num_frames * 3 + delta])
#             == num_frames
#         ]
#         symbol_date_sets = [
#             (
#                 [f"{dataset_path}/{symbol1}-{f}.npy" for f in f1 + f2],
#                 [f"{dataset_path}/{symbol2}-{f}.npy" for f in f2 + f3],
#             )
#             for f1, f2, f3 in date_set
#         ]
#         full_sets.extend(symbol_date_sets)
#     return full_sets


def spatiotemporal_dates(
    dataset_path,
    symbols,
    dates,
    num_frames=2,
    stride=1,
    max_delta=3,
    num_sets=10,
):

    date_sets = []
    deltas = np.random.choice(
        np.arange(max_delta),
        p=np.arange(max_delta, 0, -1) / sum(np.arange(max_delta, 0, -1)),
        size=num_sets,
    )
    for symbol, delta in zip(symbols[:num_sets], deltas):
        if len(dates) <= num_frames * 2 + delta:
            logging.warning(
                f"num_frames({num_frames}) and delta({delta}): {num_frames * 2 + delta}d sample >> {len(dates)}d dataset"
            )
        date_set = [
            (
                dates[x : x + num_frames],
                dates[x + num_frames + delta : x + num_frames * 2 + delta],
            )
            for x in range(0, len(dates), stride)
            if len(dates[x + num_frames + delta : x + num_frames * 2 + delta])
            == num_frames
        ]
        symbol_date_sets = [
            (
                [f"{dataset_path}/{symbol}-{f}.npy" for f in f1],
                [f"{dataset_path}/{symbol}-{f}.npy" for f in f2],
            )
            for f1, f2 in date_set
        ]
        date_sets.extend(symbol_date_sets)
    return date_sets


def load_volume(slice_filepaths):
    uneven_volume = []
    for slice_filepath in slice_filepaths:
        slice = np.load(slice_filepath)
        uneven_volume.append(slice)
    return uneven_volume


def pad_volume(uneven_volume):
    slice_shapes = np.array([slice.shape for slice in uneven_volume])
    pad_channels = np.array(slice_shapes)[:, 0].max()
    pad_strikes = np.array(slice_shapes)[:, 1].max()
    pad_exps = np.array(slice_shapes)[:, 2].max()
    pad_types = np.array(slice_shapes)[:, 3].max()
    padded_slice = np.zeros((pad_channels, pad_strikes, pad_exps, pad_types))

    volume = []
    for slice in uneven_volume:
        padded_slice[
            : slice.shape[0],
            : slice.shape[1],
            : slice.shape[2],
            : slice.shape[3],
        ] = slice[:pad_channels, :pad_strikes, :pad_exps, :pad_types]
        volume.append(padded_slice)
    volume = np.stack(volume)
    return torch.Tensor(volume)
    # TODO ensure tis still better than:
    # https://github.com/pytorch/pytorch/issues/39842
    # return torch.Tensor(volume)


def load_and_pad_volume(slice_filepaths):
    uneven_volume = load_volume(slice_filepaths)
    return pad_volume(uneven_volume)


class OcContrastiveDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        num_frames,
        pretext_task="spatiotemporal",
    ):

        symbols, dates = data.load_symbols_dates_from_dataset(dataset_path)
        self.num_frames = num_frames

        # pull dates
        self.mcal_dates = yf_utils.get_mcal_dates(
            start_date=min(dates),
            end_date=max(dates),
        )

        # spatiotemporal: https://arxiv.org/pdf/2008.03800v4.pdf
        if pretext_task == "spatiotemporal":
            self.pairs = spatiotemporal_dates(
                dataset_path=dataset_path,
                symbols=symbols,
                dates=self.mcal_dates,
                num_frames=self.num_frames,
                stride=1,
                max_delta=1,
                num_sets=10,
            )
        else:
            raise ("no pretext task set!")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return {
            "train_data": [
                load_and_pad_volume(self.pairs[idx][0]),
                load_and_pad_volume(self.pairs[idx][1]),
            ],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--pretext_task", type=str, default="spatiotemporal")
    parser.add_argument("--num_days_per_sample", type=int, default=3)
    args = parser.parse_args()

    dataset = OcContrastiveDataset(
        args.dataset_path,
        args.num_days_per_sample,
        args.pretext_task,
    )
    train_dl = DataLoader(
        dataset,
        # NOTE can be nondeterministic..
        # https://pytorch.org/docs/stable/notes/randomness.html
        collate_fn=PadCollate(
            num_frames=args.num_days_per_sample,
        ),
        batch_size=4,
    )
    batch = next(iter(train_dl))
    batch


# class CIFAR10_contrastiveGAN_DataModule(LightningDataModule):
#     def __init__(self, data_config: DataInput):
#         super().__init__()
#         self.data_config = data_config

#     def prepare_data(self):
#         # download
#         CIFAR10(self.data_config.data_dir, train=True, download=True)
#         CIFAR10(self.data_config.data_dir, train=False, download=True)

#     def setup(self, stage=None):
#         # Assign train/val datasets for use in dataloaders
#         if stage == "fit":
#             # TODO: datasets should use the same train!!!
#             # GAN data
#             self.gan_train_data = CIFAR10(
#                 self.data_config.data_dir, train=True, transform=GAN_TRAIN_TRANSFORMS
#             )
#             self.dims = tuple(self.gan_train_data[0][0].shape)
#             # Contrastive data
#             all_data = CIFAR10PairDataset(
#                 root=self.data_config.data_dir,
#                 train=True,
#                 transform=CONTRASTIVE_TRAIN_TRANSFORM,
#             )
#             self.contrastive_train_data, self.contrastive_val_data = random_split(
#                 all_data, [45000, 5000]
#             )

#         # Assign test dataset for use in dataloader(s)
#         if stage == "test" or stage is None:
#             self.gan_test_data = CIFAR10(
#                 self.data_config.data_dir, train=False, transform=GAN_TRAIN_TRANSFORMS
#             )
#             self.contrastive_test_data = CIFAR10PairDataset(
#                 root=self.data_config.data_dir,
#                 train=False,
#                 transform=CONTRASTIVE_TEST_TRANSFORM,
#             )
#             self.dims = tuple(self.gan_test_data[0][0].shape)

#     def train_dataloader(self):
#         GAN_loader = DataLoader(
#             self.gan_train_data,
#             batch_size=self.data_config.batch_size,
#             shuffle=True,
#             num_workers=self.data_config.num_workers,
#         )

#         contrastive_loader = DataLoader(
#             self.contrastive_train_data,
#             batch_size=self.data_config.batch_size,
#             shuffle=True,
#             num_workers=self.data_config.num_workers,
#         )

#         loaders = {"GAN": GAN_loader, "contrastive": contrastive_loader}
#         return loaders

#     def val_dataloader(self):
#         return DataLoader(
#             self.contrastive_val_data,
#             batch_size=self.data_config.batch_size,
#             shuffle=False,
#             num_workers=self.data_config.num_workers,
#         )

#     def test_dataloader(self):
#         GAN_loader = DataLoader(
#             self.gan_test_data,
#             batch_size=self.data_config.batch_size,
#             shuffle=False,
#             num_workers=self.data_config.num_workers,
#         )

#         contrastive_loader = DataLoader(
#             self.contrastive_test_data,
#             batch_size=self.data_config.batch_size,
#             shuffle=False,
#             num_workers=self.data_config.num_workers,
#         )

#         loaders = {"GAN": GAN_loader, "contrastive": contrastive_loader}
#         return loaders