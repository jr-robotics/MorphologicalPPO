from typing import Any, List
from abc import ABC, abstractmethod
from gymnasium.spaces import Box
import numpy as np

class Parameter(ABC):    
    @abstractmethod
    def __call__(self):
        pass
        

class ParameterList(Parameter):
    def __init__(self, value: List[Any]):
        super().__init__()
        self.value = value
        self.idx = 0
        
    def __len__(self):
        return len(self.value)

    def __call__(self):
        value = self.value[self.idx]
        self.idx = (self.idx + 1) % len(self.value)
        return value
    

class RandomParameter(Parameter, Box):
    def __init__(
        self,
        low: type[np.floating[Any]] | type[np.integer[Any]],
        high: type[np.floating[Any]] | type[np.integer[Any]], 
        shape = (1,), 
        seed: int | np.random.Generator | None = None,   
        ):
        
        dtype = type(low)
        assert type(low) == type(high)
        super().__init__(low=low, high=high, shape=shape, dtype=dtype, seed=seed)
        
    def __call__(self):
        return self.sample().item()
 

    