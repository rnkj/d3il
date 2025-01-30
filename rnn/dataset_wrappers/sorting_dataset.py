from typing import Tuple

import torch
from torch import Tensor

from environments.dataset.base_dataset import TrajectoryDataset
from environments.dataset.sorting_dataset import Sorting_Img_Dataset


_items_t = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


class Sorting_Img_Dataset_Wrapper(TrajectoryDataset):
    def __init__(self, dataset: Sorting_Img_Dataset, skip: int = 1):
        bp_img_size = dataset.bp_cam_imgs[0].size()[1:]
        inhand_img_size = dataset.inhand_cam_imgs[0].size()[1:]
        for i in range(dataset.num_data):
            bp_img = dataset.bp_cam_imgs[i]
            inhand_img = dataset.inhand_cam_imgs[i]
            valid_len = len(bp_img)

            pad_bp_img = torch.zeros(
                (dataset.max_len_data,) + bp_img_size,
                dtype=bp_img.dtype,
                device=bp_img.device,
            )
            pad_bp_img[:valid_len] = bp_img
            pad_bp_img[valid_len:] = bp_img[-1]
            dataset.bp_cam_imgs[i] = pad_bp_img

            pad_inhand_img = torch.zeros(
                (dataset.max_len_data,) + inhand_img_size,
                dtype=inhand_img.dtype,
                device=inhand_img.device,
            )
            pad_inhand_img[:valid_len] = inhand_img
            pad_inhand_img[valid_len:] = inhand_img[-1]
            dataset.inhand_cam_imgs[i] = pad_inhand_img

        self.dataset = dataset
        self.skip = skip

        assert dataset.max_len_data % skip == 0, (
            f"Maximum episode length ({dataset.max_len_data}) must be devided skip ({skip})"
        )

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(idx)
    
    def get_all_actions(self):
        return self.dataset.get_all_actions()
    
    def get_all_observations(self):
        return self.dataset.get_all_actions()

    def __len__(self) -> int:
        return self.dataset.num_data

    def __getitem__(self, index: int) -> _items_t:
        obs = self.dataset.observations[index, :: self.skip]
        act = self.dataset.actions[index, :: self.skip]
        mask = self.dataset.masks[index, :: self.skip]

        bp_imgs = self.dataset.bp_cam_imgs[index][:: self.skip]
        inhand_imgs = self.dataset.inhand_cam_imgs[index][:: self.skip]

        return (bp_imgs, inhand_imgs, obs, act, mask)
