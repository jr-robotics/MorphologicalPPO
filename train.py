import hydra
from omegaconf import DictConfig
from src.utils.utils import seed_erverything
from src.utils.instantiate import instantiate_callbacks


import os
os.environ["HYDRA_FULL_ERROR"] = "1"



@hydra.main(config_path="configs", config_name="train.yaml", version_base="1.3")
def main(cfg: DictConfig):
    
    # register environment
    hydra.utils.call(cfg.env.register)
    
    if "seed" in cfg and cfg.seed is not None:
        seed_erverything(seed=cfg.seed)
    
    agent = hydra.utils.instantiate(cfg.agent)
    callbacks = instantiate_callbacks(cfg.callbacks) if "callbacks" in cfg else None
    
    return hydra.utils.call(cfg.learner, agent=agent, callback=callbacks)
    
    
    
if __name__ == "__main__":
    main()





