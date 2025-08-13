
from gymnasium import spaces
import numpy as np

def get_obs_shape(observation_space: spaces.Space):
        
    unique_keys = set()
    for space in observation_space.spaces:
        unique_keys.update(space.spaces.keys())

    obs_shape = {key:[] for key in unique_keys}
        
    for space in observation_space.spaces:
        for key in unique_keys:
            obs_shape[key].append(space.spaces[key].shape)
            
    return obs_shape
            
                
def get_action_dim(action_space: spaces.Space):
    return [int(np.prod(space.shape)) for space in action_space.spaces]
    