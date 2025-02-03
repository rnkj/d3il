import logging
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
torch.cuda.empty_cache()


def get_working_dirs():
    from hydra.core.hydra_config import HydraConfig

    cfg = HydraConfig.get()
    config_path = Path(
        [
            path["path"]
            for path in cfg.runtime.config_sources
            if path["schema"] == "file"
        ][0]
    )

    working_dirs = sorted(
        [str(p) for p in config_path.glob("*") if p.is_dir()]
    )

    return working_dirs


@hydra.main(config_path="configs", config_name="avoiding_config.yaml")
def main(cfg: DictConfig) -> None:
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # init wandb logger and config from hydra path
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode="disabled",
        config=wandb.config,
    )

    agent = hydra.utils.instantiate(cfg.agents)

    # TODO: insert agent.load_pretrained_model() here with relative path
    working_dirs = get_working_dirs()
    agent.load_pretrained_model(working_dirs[0], sv_name=agent.eval_model_name)

    # env_sim = hydra.utils.instantiate(cfg.simulation)
    # env_sim.test_agent(agent)

    log.info("done")

    wandb.finish()


if __name__ == "__main__":
    main()
