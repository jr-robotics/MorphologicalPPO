import numpy as np
import torch
from gymnasium import spaces
from torch_geometric.nn import global_mean_pool, global_add_pool

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.vec_env.patch_gym import _patch_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecTransposeImage,
    is_vecenv_wrapped,
)


from src.common.buffers import FlexSizeRolloutBuffer
from src.common.utils import pool_actions



class FlexOnPolicyMixin():
    
    @staticmethod
    def _wrap_env(env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
        """ "
        Wrap environment with the appropriate wrappers if needed.
        For instance, to have a vectorized environment
        or to re-order the image channels.

        :param env:
        :param verbose: Verbosity level: 0 for no output, 1 for indicating wrappers used
        :param monitor_wrapper: Whether to wrap the env in a ``Monitor`` when possible.
        :return: The wrapped environment.
        """
        if not isinstance(env, VecEnv):
            # Patch to support gym 0.21/0.26 and gymnasium
            env = _patch_env(env)
            if not is_wrapped(env, Monitor) and monitor_wrapper:
                if verbose >= 1:
                    print("Wrapping the env with a `Monitor` wrapper")
                env = Monitor(env)
            if verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

        # Make sure that dict-spaces are not nested (not supported)
        # check_for_nested_spaces(env.observation_space)
        # TODO: check if fixable

        if not is_vecenv_wrapped(env, VecTransposeImage):
            wrap_with_vectranspose = False
            if isinstance(env.observation_space, spaces.Dict):
                # If even one of the keys is a image-space in need of transpose, apply transpose
                # If the image spaces are not consistent (for instance one is channel first,
                # the other channel last), VecTransposeImage will throw an error
                for space in env.observation_space.spaces.values():
                    wrap_with_vectranspose = wrap_with_vectranspose or (
                        is_image_space(space) and not is_image_space_channels_first(space)  # type: ignore[arg-type]
                    )
            else:
                wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
                    env.observation_space  # type: ignore[arg-type]
                )

            if wrap_with_vectranspose:
                if verbose >= 1:
                    print("Wrapping the env in a VecTransposeImage.")
                env = VecTransposeImage(env)

        return env
    
    
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: FlexSizeRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
          
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
                        
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                obs_dict = self.policy.batch_observations(obs=self._last_obs)
                actions, values, log_probs, _,  = self.policy(obs_dict) # FIXXME - just for debug output
                
            actions = actions.cpu().numpy()
            
            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space.common_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)


            # Convert batched data to indivdual and aggregate values and log_probs
            clipped_actions = pool_actions(actions=clipped_actions, batch=obs_dict["batch"].cpu())            
            #values =  debatch_values(values=values, batch=obs_dict["batch"])
            values =  global_mean_pool(x=values, batch=obs_dict["batch"])
            

            log_probs = global_add_pool(x=log_probs, batch=obs_dict["batch"])
            #log_probs = debatch_log_probs(log_probs=log_probs, batch=obs_dict["batch"])
            #log_probs = debatch_actions(actions=log_probs, batch=obs_dict["batch"])

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1



            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    
                    #terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    terminal_obs = infos[idx]["terminal_observation"]
                    terminal_obs_dict = self.policy.batch_observations(obs=terminal_obs)
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(obs=terminal_obs_dict)  # type: ignore[arg-type]
                    terminal_value = global_mean_pool(x=terminal_value, batch=terminal_obs_dict["batch"])
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                obs=self._last_obs,  # type: ignore[arg-type]
                action=pool_actions(actions=actions, batch=obs_dict["batch"].cpu()),
                reward=rewards,
                episode_start=self._last_episode_starts,  # type: ignore[arg-type]
                value=values,
                log_prob=log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones


        # Compute value for the last timestep
        new_obs_dict = self.policy.batch_observations(obs=new_obs)
        with torch.no_grad():
            #values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]
            values = self.policy.predict_values(obs=new_obs_dict)
            
        values = global_mean_pool(x=values, batch=new_obs_dict["batch"])
            
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

