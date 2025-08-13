from typing import List, Callable
import hydra
from src.envs.common.parameter import Parameter
from omegaconf import DictConfig
from stable_baselines3.common.callbacks import BaseCallback
from copy import deepcopy


class ParameterWrapper:    
    
    num_instantiated: int = 0
    period: int = 0
    
    
    @staticmethod
    def instantiate_parameterized(obj: Callable, **kwargs):
        processed_kwargs = ParameterWrapper._process_kwargs(deepcopy(kwargs))
        ParameterWrapper.num_instantiated = \
            (ParameterWrapper.num_instantiated + 1) % ParameterWrapper.period
        return obj(**processed_kwargs)


    @staticmethod
    def _process_kwargs(kwargs: dict):

        for key in kwargs.keys():
            if isinstance(kwargs[key], Parameter):
                assert ParameterWrapper.period == 0 or \
                    (ParameterWrapper.period > 0 and len(kwargs[key]) == ParameterWrapper.period)
                    
                ParameterWrapper.period = len(kwargs[key])
                
                kwargs[key] = kwargs[key].sample(ParameterWrapper.num_instantiated)
                
            elif isinstance(kwargs[key], dict):
                # Recursively process the dictionary without instantiating the object
                kwargs[key] = ParameterWrapper._process_kwargs(kwargs[key])

    
        return kwargs
    
    



def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[BaseCallback]:
    """ Instantiates callbacks from config. """
    callbacks: List[BaseCallback] = []

    if not callbacks_cfg:
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return list(callbacks)
