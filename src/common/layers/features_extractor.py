from typing import Tuple
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from src.envs.spaces import AugmentedKinematicGraph	


class AugmentedGraphExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: AugmentedKinematicGraph | Tuple[AugmentedKinematicGraph],
        features_dim: int,
        model: nn.Module, 
        ):
        
        super().__init__(observation_space=observation_space, features_dim=features_dim)
        
        self.model = model(
            global_shape=observation_space["global_space"].shape,
            node_shape=observation_space["node_space"].shape,
            edge_shape=observation_space["edge_space"].shape,
            features_dim=features_dim,
        )
       
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

        

        
