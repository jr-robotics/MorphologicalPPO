from typing import Dict, Tuple, Union, List
import numpy as np
import torch
from torch_geometric.data import Data, Batch



def pool_actions(actions: Union[torch.Tensor, np.ndarray], batch: torch.Tensor) -> List[np.ndarray]:
    actions = actions.squeeze()
    return [actions[batch==idx] for idx in batch.unique()]

    
def dict_to_graph(data: Dict):
    return Data(
        x=data["node_space"],
        edge_index=data["edge_indices"].t(),
        edge_attr=data["edge_space"],
    )
 

def batch_to_dict(batch: Batch, **kwargs):
    return dict(
        node_space=batch.x,
        edge_space=batch.edge_attr,
        edge_indices=batch.edge_index,
        **{key: value for key, value in batch.items() if key not in ["x", "edge_attr", "edge_index", "ptr"]},
        **kwargs,
    )
 
def batch_vec_obs(vec_obs: Tuple[Dict], edge_batch: bool=True):
    graphs = [dict_to_graph(data) for data in vec_obs]
    batch = []
    for idx, graph in enumerate(graphs):
        batch = np.append(batch, idx*np.ones(len(graph.edge_attr)))

    data_batch = Batch.from_data_list(graphs)
    if edge_batch:
        data_batch.batch = torch.from_numpy(batch).long().to(data_batch.x.device)
    return data_batch



