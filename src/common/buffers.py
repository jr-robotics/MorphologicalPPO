from typing import Dict, Generator, List, Optional, Union, Sequence, NamedTuple
import torch
import numpy as np
import torch as th
from gymnasium import spaces


from stable_baselines3.common.vec_env import VecNormalize
from src.common.preprocessing import get_obs_shape, get_action_dim
from stable_baselines3.common.buffers import RolloutBuffer, DictRolloutBuffer
from stable_baselines3.common.type_aliases import TensorDict


try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None



class FlexSizeRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class FlexSizeRolloutBuffer(DictRolloutBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: list
    actions: list
    #log_probs: list
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        
        super(RolloutBuffer, self).__init__(
            buffer_size,
            observation_space.cat_space,
            action_space.cat_space,
            device,
            n_envs=n_envs
        )
    
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]
        self.action_dim = get_action_dim(action_space)
        self.reset()


    def get_buffered_action_dims(self) -> List[int]:
        return [len(action) for action in self.actions]
        

    def reset(self) -> None:    
        self.observations = {key: [[] for _ in range(self.buffer_size*self.n_envs)] \
            for key in self.observation_space.keys()}
        
        self.actions = [[] for _ in range(self.buffer_size*self.n_envs)]    
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        super(RolloutBuffer, self).reset()
                
                
    def add(  # type: ignore[override]
        self,
        obs: Union[Dict[str, np.ndarray], Sequence[Dict[str, np.ndarray]]],
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        
        if not isinstance(obs, Sequence):
            obs = [get_obs_shape]

        if not isinstance(action, Sequence):
            action = [action]
        

        for n, (nth_obs, nth_action, nth_log_prob) in enumerate(zip(obs,action,log_prob)):
            self.observations[self.buffer_size*n + self.pos] = nth_obs                    
            self.actions[self.buffer_size*n + self.pos] = np.array(nth_action)

        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy().flatten()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True


    def _convert_batch_index(self, flat_idx: int):
        """
        Convert a flattened buffer index back into the corresponding
        buffer list index and environment index.

        Args:
            flat_idx (int): The index after swap_and_flatten (e.g., in [0, buffer_size * n_envs))

        Returns:
            list_idx (int): Corresponding index into the buffer (time step)
            env_idx (int): Corresponding environment index
        """
        env_idx = flat_idx // self.buffer_size
        list_idx = flat_idx % self.buffer_size
        return list_idx, env_idx



    def get(  # type: ignore[override]
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[FlexSizeRolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        # Prepare the data
        if not self.generator_ready:
            _tensor_names = ["values", "advantages", "returns", "log_probs"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            
            self.generator_ready = True
            
            
        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size


    def _get_samples(  # type: ignore[override]
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> FlexSizeRolloutBufferSamples:
                
        return FlexSizeRolloutBufferSamples(
            observations=[self.observations[idx] for idx in batch_inds],
            actions=[self.actions[idx] for idx in batch_inds], 
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
        )