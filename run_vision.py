import os
import logging
import random

import hydra
import numpy as np
from tqdm import tqdm

import wandb
from omegaconf import DictConfig, OmegaConf
import torch


log = logging.getLogger(__name__)


OmegaConf.register_new_resolver(
     "add", lambda *numbers: sum(numbers)
)
torch.cuda.empty_cache()


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="configs", config_name="aligning_vision_config.yaml")
def main(cfg: DictConfig) -> None:

    # if cfg.seed in [0, 1]:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # elif cfg.seed in [2, 3]:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # elif cfg.seed in [4, 5]:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    set_seed_everywhere(cfg.seed)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.group,
        mode=cfg.wandb.mode,
        config=wandb.config
    )

    agent = hydra.utils.instantiate(cfg.agents)
    agent.train_vision_agent()
    log.info("done")

    wandb.finish()


if __name__ == "__main__":
    main()