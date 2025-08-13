from typing import Dict
import gymnasium as gym
from gymnasium import register, make
from omegaconf import DictConfig
from pydoc import locate

def remove_special_keys_from_config(cfg: DictConfig) -> Dict:
    clear_dict = {
        k:v for k,v in cfg.items() if not (k.startswith('_') and k.endswith('_'))
    }
    return clear_dict
    
    
def register_from_config(cfg: DictConfig) -> str:
    env_identifier = cfg.env._target_.split(".")[-1]
    gym.register(
        id=env_identifier,
        entry_point=locate(cfg.env._target_))
    return env_identifier

