import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple, Optional, Sequence, Type

import gymnasium as gym
import numpy as np

import cloudpickle
import multiprocessing as mp


from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv

from src.envs.spaces import AugmentedKinematicGraph, KinematicGraph, GraphVectorSpace



class GraphVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``Cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    :raises ValueError: If the same environment instance is passed as the output of two or more different env_fn.
    """

    actions: np.ndarray

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [_patch_env(fn()) for fn in env_fns]
        if len(set([id(env.unwrapped) for env in self.envs])) != len(self.envs):
            raise ValueError(
                "You tried to create multiple environments, but the function to create them returned the same instance "
                "instead of creating different objects. "
                "You are probably using `make_vec_env(lambda: env)` or `DummyVecEnv([lambda: env] * n_envs)`. "
                "You should replace `lambda: env` by a `make_env` function that "
                "creates a new instance of the environment at every call "
                "(using `gym.make()` for instance). You can take a look at the documentation for an example. "
                "Please read https://github.com/DLR-RM/stable-baselines3/issues/1151 for more information."
            )
            
            
        env = self.envs[0]
        super().__init__(
            num_envs=len(env_fns),
            observation_space=GraphVectorSpace(spaces=[env.observation_space for env in self.envs]),
            action_space=GraphVectorSpace(spaces=[env.action_space for env in self.envs]),
        )
            

        
        self._setup_obs_buffer()
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.metadata = env.metadata
        


    def _setup_obs_buffer(self):
        self.buf_obs = [] 
        for env in self.envs:
            obs_space = env.observation_space
            assert isinstance(obs_space, KinematicGraph)
            
            buf_obs = dict(
                node_obs=np.zeros(shape=obs_space.node_space.shape, dtype=obs_space.node_space.dtype),
                edge_obs=np.zeros(shape=obs_space.edge_space.shape, dtype=obs_space.edge_space.dtype),
                edge_indices=obs_space.edge_indices,
            )
            
            if isinstance(obs_space, AugmentedKinematicGraph):
                buf_obs["global_obs"] = np.zeros(shape=(obs_space.global_space.shape), dtype=obs_space.global_space.dtype),

            self.buf_obs.append(buf_obs)



    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        # Avoid circular imports
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            # convert to SB3 VecEnv api
            self.buf_dones[env_idx] = terminated or truncated
            # See https://github.com/openai/gym/issues/3102
            # Gym 0.26 introduces a breaking change
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def reset(self) -> VecEnvObs:
        
        vec_obs = []
        
        for env_idx in range(self.num_envs):
            maybe_options = {"options": self._options[env_idx]} if self._options[env_idx] else {}
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(seed=self._seeds[env_idx], **maybe_options)
            vec_obs.append(obs)
            
        for env_idx, obs in enumerate(vec_obs):
            self._save_obs(env_idx, obs)
        
        # Seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return self._obs_from_buf()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[Optional[np.ndarray]]:
        if self.render_mode != "rgb_array":
            warnings.warn(
                f"The render mode is {self.render_mode}, but this method assumes it is `rgb_array` to obtain images."
            )
            return [None for _ in self.envs]
        return [env.render() for env in self.envs]  # type: ignore[misc]

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.

        :param mode: The rendering type.
        """
        return super().render(mode=mode)
            
            
    def _save_obs(self, env_idx: int, obs: Dict) -> None:
        self.buf_obs[env_idx] = obs

    def _obs_from_buf(self) -> Dict:
        return deepcopy(self.buf_obs)
        

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [env_i.get_wrapper_attr(attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [env_i.get_wrapper_attr(method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]


class SubprocGraphVecEnv(VecEnv):
    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in range(n_envs)])
        self.processes = []

        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, cloudpickle.dumps(env_fn))
            proc = mp.Process(target=self._worker, args=args, daemon=True)
            proc.start()
            work_remote.close()
            self.processes.append(proc)

        # Get observation and action spaces
        observation_space = [[] for _ in range(n_envs)]
        action_space = [[] for _ in range(n_envs)]
        
        for idx, remote in enumerate(self.remotes):
            remote.send(("get_spaces", None))
            observation_space[idx], action_space[idx] = remote.recv()

        super().__init__(
            num_envs=n_envs,
            observation_space=GraphVectorSpace(spaces=observation_space),
            action_space=GraphVectorSpace(spaces=action_space)
        )

        self.buf_obs = None
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_infos = [{} for _ in range(self.num_envs)]

    def _worker(self, remote, parent_remote, env_fn_wrapper):
        parent_remote.close()
        env_fn = cloudpickle.loads(env_fn_wrapper)
        env = env_fn()
        try:
            while True:
                cmd, data = remote.recv()
                if cmd == "step":
                    obs, reward, terminated, truncated, info = env.step(data)
                    done = terminated or truncated
                    if done:
                        info["terminal_observation"] = obs
                        obs, _ = env.reset()
                    remote.send((obs, reward, done, info))
                elif cmd == "reset":
                    obs, info = env.reset()
                    remote.send((obs, info))
                elif cmd == "close":
                    env.close()
                    remote.close()
                    break
                elif cmd == "get_spaces":
                    remote.send((env.observation_space, env.action_space))
                elif cmd == "render":
                    remote.send(env.render())
                elif cmd == "get_attr":
                    remote.send(getattr(env, data))
                elif cmd == "set_attr":
                    attr_name, value = data
                    setattr(env, attr_name, value)
                    remote.send(None)
                elif cmd == "env_method":
                    method_name, method_args, method_kwargs = data
                    method = getattr(env, method_name)
                    remote.send(method(*method_args, **method_kwargs))
                elif cmd == "is_wrapped":
                    wrapper_class_name = data
                    from stable_baselines3.common import env_util
                    result = env_util.is_wrapped(env, wrapper_class_name)
                    remote.send(result)
                else:
                    raise NotImplementedError(f"Unknown command: {cmd}")
        except KeyboardInterrupt:
            print("Worker: caught KeyboardInterrupt")

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        self.buf_obs = list(obs)
        self.buf_rews[:] = rews
        self.buf_dones[:] = dones
        self.buf_infos = list(infos)
        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos)

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        obs, infos = zip(*results)
        self.buf_obs = list(obs)
        return self._obs_from_buf()

    def _obs_from_buf(self):
        return deepcopy(self.buf_obs)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for proc in self.processes:
            proc.join()
        self.closed = True

    def render(self, mode: Optional[str] = None):
        for remote in self.remotes:
            remote.send(("render", None))
        return [remote.recv() for remote in self.remotes]

    def get_images(self):
        return self.render(mode="rgb_array")

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("get_attr", attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("set_attr", (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(("env_method", (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        target_remotes = self._get_target_remotes(indices)
        wrapper_class_name = wrapper_class.__name__
        for remote in target_remotes:
            remote.send(("is_wrapped", wrapper_class_name))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices: VecEnvIndices) -> List[Any]:
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]