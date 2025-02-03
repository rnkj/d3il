import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional

import hydra
import numpy as np
import torch
import tyro
import wandb
from omegaconf import OmegaConf


log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
torch.cuda.empty_cache()


@dataclass
class SimulationArgs:
    # random seed
    seed: Optional[int] = None
    # CPU/CUDA device
    device: str = "cuda"
    # render or not
    render: bool = False
    # number of CPU cores
    n_cores: Optional[int] = None
    # number of contexts of trained tasks
    n_contexts: Optional[int] = None
    # number of trajectory tested per contexts
    n_trajectories_per_context: Optional[int] = None


@dataclass
class Args:
    # simulation option
    simulation: SimulationArgs

    # log directory where checkpoints and config files are saved
    logdir: str

    # index of trial
    multirun_index: int = 0

    # model type: "eval" or "last"
    model_type: Literal["eval", "last"] = "eval"


def main(args: Args) -> None:
    # search multirun directories
    multirun_dirs = sorted(set([p.parent for p in Path(args.logdir).glob("**/*.pth")]))
    target_dir = multirun_dirs[args.multirun_index]

    # get config file of training
    config_file = os.path.join(target_dir, ".hydra/config.yaml")
    cfg = OmegaConf.load(config_file)

    # set custom parameters of simulation
    if args.simulation.seed is not None:
        cfg.simulation.seed = args.simulation.seed
    cfg.simulation.device = args.simulation.device
    cfg.simulation.render = args.simulation.render
    if args.simulation.n_cores is not None:
        cfg.simulation.n_cores = args.simulation.n_cores
    if args.simulation.n_contexts is not None:
        cfg.simulation.n_contexts = args.simulation.n_contexts
    if args.simulation.n_trajectories_per_context is not None:
        cfg.simulation.n_trajectories_per_context = (
            args.simulation.n_trajectories_per_context
        )

    np.random.seed(cfg.simulation.seed)
    torch.manual_seed(cfg.simulation.seed)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode="disabled",
        config=wandb.config,
    )

    agent = hydra.utils.instantiate(cfg.agents)

    if args.model_type == "eval":
        sv_name = agent.eval_model_name
    elif args.model_type == "last":
        sv_name = agent.last_model_name
    else:
        raise ValueError

    agent.load_pretrained_model(target_dir, sv_name=sv_name)

    env_sim = hydra.utils.instantiate(cfg.simulation)
    env_sim.test_agent(agent)

    log.info("done")

    wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
