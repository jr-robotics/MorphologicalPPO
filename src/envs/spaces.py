from typing import Any, Optional, Union, Sequence, Tuple, NamedTuple, Iterable, KeysView
from gymnasium.spaces import Box, Space
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import Tuple as TupleSpace


from matplotlib.pylab import Generator
import numpy as np
from numpy.typing import NDArray



def get_common_shape(shapes):
    
    dims = set([len(shape) for shape in shapes])
    assert len(dims) == 1, "All shapes must have the same number of dimensions."
    dims = dims.pop()
    
    if dims == 1:
        return shapes[0] if len(set(shapes)) == 1 else (1,) 
    else:
        first_shape = shapes[0]
        assert all(len(shape) == len(first_shape) for shape in shapes), \
            "All shapes must have the same number of dimensions."
        
        common_shape = []
        for i in range(len(first_shape)):
            unique_dims = set(shape[i] for shape in shapes)
            if len(unique_dims) == 1:
                common_shape.append(unique_dims.pop())
        
    return tuple(common_shape)


def get_stacked_shape(shapes):
    if not shapes:
        return ()
    
    shapes = [shape if isinstance(shape, tuple) else (shape,) for shape in shapes]
    dims = set(len(shape) for shape in shapes)
    
    assert len(dims) == 1, "All shapes must have the same number of dimensions."
    dims = dims.pop()
    
    stacked_dim = sum(shape[0] for shape in shapes)
    
    if dims == 1:
        return (stacked_dim,)
    else:
        return (stacked_dim,) + shapes[0][1:]



class EdgeIndices():
    def __init__(self, src: Sequence[int], dest: Sequence[int], dtype=np.uint32, **kwargs):
        self.indices = np.array([src, dest], dtype=dtype, **kwargs)
        self.unique_nodes = np.unique(self.indices)
        
    @property
    def num_nodes(self):
        return len(self.unique_nodes)

    @property
    def num_edges(self):
        return self.indices.shape[1]
    
    @property
    def shape(self):
        return self.indices.shape
    
    def __call__(self) -> NDArray:
        return self.indices


class KinematicGraphInstance(NamedTuple):
    nodes: NDArray[Any]
    edges: NDArray[Any]
    edge_links: NDArray[Any]

   
        
class KinematicGraph(DictSpace):
    
    def __init__(
        self,
        node_attr_shape: Tuple[int],
        edge_attr_shape: Tuple[int],
        edge_indices: Sequence[Tuple[int, int]],
        dtype: Optional[Union[np.float32, np.float64]] = np.float64,
        seed: Optional[Union[int, Generator]] = None,
        ):
        
        assert all(np.diff(np.unique(edge_indices)) == 1)
        self._edge_indices = edge_indices

        node_shape = (*node_attr_shape, self.num_nodes)
        edge_shape = (*edge_attr_shape, self.num_edges)
        

        super().__init__(
            spaces=dict(
                node_space=Box(low=-np.inf, high=np.inf, shape=node_shape, dtype=dtype),
                edge_space=Box(low=-np.inf, high=np.inf, shape=edge_shape, dtype=dtype),
                edge_indices=Box(low=0, high=self.num_nodes-1, shape=(self.num_edges, 2), dtype=np.int64),
            ),
            seed=seed)


    @property
    def node_space(self):
        return self.spaces["node_space"]
    
    @property
    def edge_space(self):
        return self.spaces["edge_space"]

    @property
    def shape(self):
        return dict(
            node_space=self.node_space.shape,
            edge_space=self.edge_space.shape,
            edge_indices=self.spaces["edge_indices"].shape,
        )

    @property
    def edge_indices(self):
        return np.array(self._edge_indices).reshape(self.num_edges, 2)
    
    @property
    def num_nodes(self):
        return len(np.unique(self._edge_indices))
    
    @property
    def num_edges(self):
        return len(self._edge_indices)
    
    
    # def sample(self) -> dict[str, Any]:  
    #     xx = super().sample()
    #     node_attr = self.node_space.sample()
    #     edge_attr = self.edge_space.sample()
    #     return dict(nodes=node_attr, edges=edge_attr, edge_links=self.edge_indices)




class AugmentedKinematicGraph(KinematicGraph):
    def __init__(
        self,
        global_space: Box,
        node_attr_shape: Tuple[int],
        edge_attr_shape: Tuple[int],
        edge_indices: Sequence[Tuple[int, int]],
        dtype: Optional[Union[np.float32, np.float64]] = np.float64,
        seed: Optional[Union[int, Generator]] = None,
        ):
            
        assert all(np.diff(np.unique(edge_indices)) == 1)
        self._edge_indices = edge_indices
        node_shape = (self.num_nodes, *node_attr_shape)
        edge_shape = (self.num_edges, *edge_attr_shape)
        
        
        super(KinematicGraph, self).__init__(
            spaces=dict(
                global_space=global_space,
                node_space=Box(low=-np.inf, high=np.inf, shape=node_shape, dtype=dtype),
                edge_space=Box(low=-np.inf, high=np.inf, shape=edge_shape, dtype=dtype),
                edge_indices=Box(low=0, high=self.num_nodes-1, shape=(self.num_edges, 2), dtype=np.int64),
            ),
            seed=seed,
        )
        self._edge_indices = edge_indices
        
            
    @property
    def global_space(self):
        return self.spaces["global_space"]
            
    @property
    def shape(self):
        return dict(global_space=self.global_space.shape, **super().shape)

    def sample(self):
        return dict(global_obs=self.global_space.sample(), **super().sample())




class GraphVectorSpace(TupleSpace):
    def __init__(
        self,
        spaces: Iterable[Space[Any]],
        seed: int | Sequence[int] | np.random.Generator | None = None,
    ):
        
        self.space_class = type(spaces[0])
        assert all(isinstance(space, self.space_class) for space in spaces),\
           "All spaces must be of the same type" 

        super().__init__(spaces=spaces, seed=seed)
        
    @property
    def low(self):
        return self.common_space.low
    
    @property
    def high(self):
        return self.common_space.high       
        
    @property    
    def cat_space(self):        
        if isinstance(self.spaces[0], DictSpace):
            spaces = dict()

            for key in self.spaces[0].spaces.keys():
                stacked_shape = get_stacked_shape([space.spaces[key].shape for space in self.spaces])
                spaces[key] = Box(
                    low=np.unique(self[0][key].low)[0],
                    high=np.unique(self[0][key].high)[0],
                    shape=stacked_shape,
                    dtype=self[0][key].dtype
                )
            
            return DictSpace(spaces=spaces)
            
        elif isinstance(self.spaces[0], Box):
            stacked_shape = get_stacked_shape([space.shape for space in self.spaces])
            return Box(
                low=np.unique(self[0].low)[0],
                high=np.unique(self[0].high)[0],
                shape=stacked_shape,
                dtype=self[0].dtype
            )
            
    @property
    def common_space(self):
        if isinstance(self.spaces[0], DictSpace):
            spaces = dict()

            for key in self.spaces[0].spaces.keys():
                common_shape = get_common_shape([space.spaces[key].shape for space in self.spaces])
                spaces[key] = Box(
                    low=min([space[key].low.min() for space in self.spaces]),
                    high=max([space[key].high.max() for space in self.spaces]),
                    shape=common_shape,
                    dtype=self[0][key].dtype
                )
            
            return DictSpace(spaces=spaces)
            
        elif isinstance(self.spaces[0], Box):
            common_shape = get_common_shape([space.shape for space in self.spaces])
            return Box(
                low=np.unique(self[0].low)[0],
                high=np.unique(self[0].high)[0],
                shape=common_shape,
                dtype=self[0].dtype
            )
            


    def keys(self):
        return self.spaces[0].keys() if isinstance(self.spaces[0], DictSpace) else KeysView({})
        
        
        
        
                

        
        
        

if __name__ == "__main__":
    gs1 = AugmentedKinematicGraph(
        global_space=Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
        node_attr_shape = (4,),
        edge_attr_shape = (5,),
        edge_indices=[(0,1),(1,2)],     
    )
    
    gs2 = AugmentedKinematicGraph(
        global_space=Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
        node_attr_shape = (4,),
        edge_attr_shape = (5,),
        edge_indices=[(0,1),(1,2),(2,3),(2,2)],      
    )

    
    vecspace = GraphVectorSpace(
        spaces=[gs1, gs2],
        seed=1234,
    )

    
    action_space = GraphVectorSpace(
        spaces=[
            Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64),
        ]
    )
    
    action_space2 = GraphVectorSpace(
        spaces=[
            Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        ]
    )
    

    print("vecspace.cat_space.shape", vecspace.cat_space.shape)
    print("vecspace.common_space.shape", vecspace.common_space.shape)
    print("action_space.cat_space.shape", action_space.cat_space.shape)
    print("action_space.common_space.shape", action_space.common_space.shape)
    print("action_space2.cat_space.shape", action_space2.cat_space.shape)
    print("action_space2.common_space.shape", action_space2.common_space.shape)

    ret = vecspace.sample()
    pass
