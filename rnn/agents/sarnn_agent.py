import logging
import os
from typing import Dict, Optional, Tuple

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from eipl.utils import LossScheduler
from omegaconf import DictConfig
from torch import Tensor

from agents.base_agent import BaseAgent
from rnn.models.sarnn import SARNN

from IPython import embed as e

log = logging.getLogger(__name__)

_DEFAULT_LOSS_WEIGHTS: Dict[str, float] = {
    "image": 1.0,
    "action": 1.0,
    "attention": 1.0,
}


class SARNN_Agent(BaseAgent):
    model: SARNN

    def __init__(
        self,
        model: DictConfig,
        optimization: DictConfig,
        trainset: DictConfig,
        valset: DictConfig,
        train_batch_size,
        val_batch_size,
        num_workers,
        device: str,
        epoch: int,
        scale_data,
        eval_every_n_epochs: int = 50,
        loss_weights: Dict[str, float] = _DEFAULT_LOSS_WEIGHTS,
    ):
        super().__init__(
            model=model,
            trainset=trainset,
            valset=valset,
            train_batch_size=train_batch_size,
            val_batch_size=val_batch_size,
            num_workers=num_workers,
            device=device,
            epoch=epoch,
            scale_data=scale_data,
            eval_every_n_epochs=eval_every_n_epochs,
        )

        # Define the number of GPUs available
        num_gpus = torch.cuda.device_count()

        # Check if multiple GPUs are available and select the appropriate device
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for training.")
            self.model = nn.DataParallel(self.model)

        self.optimizer = hydra.utils.instantiate(
            optimization, params=self.model.parameters()
        )

        self.eval_model_name = "eval_best_sarnn.pth"
        self.last_model_name = "last_sarnn.pth"

        self.min_action = torch.from_numpy(self.scaler.y_bounds[0, :]).to(
            self.device
        )
        self.max_action = torch.from_numpy(self.scaler.y_bounds[1, :]).to(
            self.device
        )

        self.loss_weights = loss_weights.copy()
        self.pt_loss_schedulers = {
            key: LossScheduler(decay_end=1000, curve_name="s")
            for key in self.model.IMG_KEYS
        }

        self.rnn_state = None

    def train_agent(self):
        raise NotImplementedError("SARNN must be trained with images")

    def train_vision_agent(self):
        data: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

        train_loss = []
        for data in self.train_dataloader:
            bp_imgs, inhand_imgs, obs, action, mask = data

            bp_imgs = bp_imgs.to(self.device)
            inhand_imgs = inhand_imgs.to(self.device)

            obs = self.scaler.scale_input(obs)
            action = self.scaler.scale_output(action)

            state = (bp_imgs, inhand_imgs, obs)

            batch_loss = self.train_step(state, action)

            train_loss.append(batch_loss)

            wandb.log(
                {
                    "loss": batch_loss,
                }
            )

    def train_step(
        self, state, actions: torch.Tensor, goal: Optional[torch.Tensor] = None
    ):
        """
        Executes a single training step on a mini-batch of data
        """
        self.model.train()

        if goal is not None:
            # goal = self.scaler.scale_input(goal)
            # out = self.model(torch.cat([state, goal], dim=-1))
            raise NotImplementedError
        else:
            rnn_state = None
            in_bp_imgs, in_inhand_imgs, obs = state
            batch_size, episode_length = in_bp_imgs.size()[:2]

            in_imgs_dic = {"bp": in_bp_imgs, "inhand": in_inhand_imgs}

            out_imgs_dic = {
                "bp": torch.zeros_like(in_bp_imgs[:, 1:]),
                "inhand": torch.zeros_like(in_inhand_imgs[:, 1:]),
            }
            out_actions = torch.zeros_like(actions[:, 1:])
            enc_pts_dic = {
                key: torch.zeros(
                    (batch_size, episode_length, self.model.k_dim * 2)
                )
                for key in self.model.IMG_KEYS
            }
            dec_pts_dic = {
                key: torch.zeros(
                    (batch_size, episode_length, self.model.k_dim * 2)
                )
                for key in self.model.IMG_KEYS
            }

            for t in range(episode_length - 1):
                xi = {"bp": in_bp_imgs[:, t], "inhand": in_inhand_imgs[:, t]}
                out = self.model(xi, obs[:, t], rnn_state)

                out_actions[:, t] = out[1]
                for key in self.model.IMG_KEYS:
                    out_imgs_dic[key][:, t] = out[0][key]
                    enc_pts_dic[key][:, t] = out[2][key]
                    dec_pts_dic[key][:, t] = out[3][key]
                rnn_state = out[4]

        img_loss, pt_loss = (0, 0)
        img_loss_dic, pt_loss_dic = {}, {}
        for key in self.model.IMG_KEYS:
            in_img = in_imgs_dic[key]
            out_img = out_imgs_dic[key]
            img_loss_dic[key] = (
                F.mse_loss(out_img, in_img[:, 1:]) * self.loss_weights["image"]
            )
            img_loss += img_loss_dic[key]

            enc_pts = enc_pts_dic[key]
            dec_pts = dec_pts_dic[key]
            scheduler = self.pt_loss_schedulers[key]
            pt_loss_dic[key] = F.mse_loss(
                dec_pts[:, :-1], enc_pts[:, 1:]
            ) * scheduler(self.loss_weights["attention"])
            pt_loss += pt_loss_dic[key]

        act_loss = (
            F.mse_loss(out_actions, actions[:, 1:])
            * self.loss_weights["action"]
        )
        loss = act_loss + img_loss + pt_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(
        self, state, actions: torch.Tensor, goal: Optional[torch.Tensor] = None
    ):
        """
        Method for evaluating the model on one epoch of data
        """
        self.model.eval()

        if goal is not None:
            # goal = self.scaler.scale_input(goal)
            # out = self.model(torch.cat([state, goal], dim=-1))
            raise NotImplementedError
        else:
            rnn_state = None
            in_bp_imgs, in_inhand_imgs, obs = state
            batch_size, episode_length = in_bp_imgs.size()[:2]

            in_imgs_dic = {"bp": in_bp_imgs, "inhand": in_inhand_imgs}

            out_imgs_dic = {
                "bp": torch.zeros_like(in_bp_imgs[:, 1:]),
                "inhand": torch.zeros_like(in_inhand_imgs[:, 1:]),
            }
            out_actions = torch.zeros_like(actions[:, 1:])
            enc_pts_dic = {
                key: torch.zeros(
                    (batch_size, episode_length, self.model.k_dim * 2)
                )
                for key in self.model.IMG_KEYS
            }
            dec_pts_dic = {
                key: torch.zeros(
                    (batch_size, episode_length, self.model.k_dim * 2)
                )
                for key in self.model.IMG_KEYS
            }

            for t in range(episode_length - 1):
                xi = {"bp": in_bp_imgs[:, t], "inhand": in_inhand_imgs[:, t]}
                out = self.model(xi, obs[:, t], rnn_state)

                out_actions[:, t] = out[1]
                for key in self.model.IMG_KEYS:
                    out_imgs_dic[key][:, t] = out[0][key]
                    enc_pts_dic[key][:, t] = out[2][key]
                    dec_pts_dic[key][:, t] = out[3][key]
                rnn_state = out[4]

        img_loss, pt_loss = (0, 0)
        img_loss_dic, pt_loss_dic = {}, {}
        for key in self.model.IMG_KEYS:
            in_img = in_imgs_dic[key]
            out_img = out_imgs_dic[key]
            img_loss_dic[key] = (
                F.mse_loss(out_img, in_img[:, 1:]) * self.loss_weights["image"]
            )
            img_loss += img_loss_dic[key]

            enc_pts = enc_pts_dic[key]
            dec_pts = dec_pts_dic[key]
            scheduler = self.pt_loss_schedulers[key]
            pt_loss_dic[key] = F.mse_loss(
                dec_pts[:, :-1], enc_pts[:, 1:]
            ) * scheduler(self.loss_weights["attention"])
            pt_loss += pt_loss_dic[key]

        act_loss = (
            F.mse_loss(out_actions, actions[:, 1:])
            * self.loss_weights["action"]
        )
        loss = act_loss + img_loss + pt_loss

        return loss.item()

    @torch.no_grad()
    def predict(
        self, state, goal: Optional[torch.Tensor] = None, if_vision=False
    ) -> torch.Tensor:
        """
        Method for predicting one step with input data
        """
        self.model.eval()

        bp_image, inhand_image, obs = state

        bp_image = (
            torch.from_numpy(bp_image)
            .to(self.device)
            .float()
            .unsqueeze(0)
        )
        inhand_image = (
            torch.from_numpy(inhand_image)
            .to(self.device)
            .float()
            .unsqueeze(0)
        )
        obs = (
            torch.from_numpy(obs)
            .to(self.device)
            .float()
            .unsqueeze(0)
        )
        obs = self.scaler.scale_input(obs)

        if goal is not None:
            # goal = self.scaler.scale_input(goal)
            # out = self.model(torch.cat([state, goal], dim=-1))
            raise NotImplementedError
        else:
            xi = {"bp": bp_image, "inhand": inhand_image}
            pred = self.model(xi, obs, self.rnn_state)
            out = pred[1]
            self.rnn_state = pred[4]

        out = out.clamp_(self.min_action, self.max_action)

        model_pred = self.scaler.inverse_scale_output(out)
        return model_pred.detach().cpu().numpy()[0]

    def load_pretrained_model(self, weights_path: str, sv_name=None) -> None:
        """
        Method to load a pretrained model weights inside self.model
        """

        if sv_name is None:
            self.model.load_state_dict(
                torch.load(os.path.join(weights_path, "model_state_dict.pth"))
            )
        else:
            self.model.load_state_dict(
                torch.load(os.path.join(weights_path, sv_name))
            )
        log.info("Loaded pre-trained model parameters")

    def store_model_weights(self, store_path: str, sv_name=None) -> None:
        """
        Store the model weights inside the store path as model_weights.pth
        """

        if sv_name is None:
            torch.save(
                self.model.state_dict(),
                os.path.join(store_path, "model_state_dict.pth"),
            )
        else:
            torch.save(
                self.model.state_dict(), os.path.join(store_path, sv_name)
            )

    def reset(self):
        self.rnn_state = None
