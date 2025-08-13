import warnings

from typing import Type, TypeVar, Union, Optional, Dict, Any
import numpy as np
import torch
from torch.nn import functional as F

from gymnasium import spaces
from torch_geometric.nn import global_mean_pool, global_add_pool
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.utils import get_schedule_fn, explained_variance
from stable_baselines3 import PPO

from src.policies.flex_actor_critc import FlexActorCritic
from src.common.buffers import FlexSizeRolloutBuffer
from src.agents.flex_base import FlexOnPolicyMixin

SelfFlexPPO = TypeVar("SelfFlexPPO", bound="FlexPPO")

class ClippingStats(Dict):
    def __init__(self, dims: np.ndarray):
        self.dims = dims
        self.mean = {str(dim):[] for dim in self.dims}
        self.std = {str(dim):[] for dim in self.dims}
        self.max = {str(dim):[] for dim in self.dims}
        self.min = {str(dim):[] for dim in self.dims}
        self.freq = {str(dim):[] for dim in self.dims}
        self.batch_size = {str(dim):[] for dim in self.dims}           
        
        
    def __getitem__(self, key):
        return dict(
            mean = np.mean(self.mean[key]),
            std = np.mean(self.std[key]),
            max = np.max(self.max[key]),
            min = np.max(self.min[key]),
            freq = np.mean(self.freq[key]),
        )
        
    def keys(self):
        return [str(dim) for dim in sorted(self.dims)]
        
    def start_epoch(self):
        for dim in self.dims:
            key = str(dim)
            self.mean[key].append([])
            self.std[key].append([])
            self.max[key].append([])
            self.min[key].append([])
            self.freq[key].append([])
            self.batch_size[key].append([])

    def queue_batch(
            self,
            dims: np.ndarray,
            ratio: np.ndarray,
            clipped_ratio: np.ndarray) -> None:
        
        delta_ratio = ratio - clipped_ratio
        abs_clip_fraction = np.abs(delta_ratio)/ratio
        for dim in np.unique(dims):
            key = str(dim)
            self.mean[key][-1].append(abs_clip_fraction[dims==dim].mean())
            self.std[key][-1].append(abs_clip_fraction[dims==dim].std())
            self.max[key][-1].append(abs_clip_fraction[dims==dim].max())
            self.min[key][-1].append(abs_clip_fraction[dims==dim].min())
            self.freq[key][-1].append(abs_clip_fraction[dims==dim] > 0)
            self.batch_size[key][-1].append(len(abs_clip_fraction[dims==dim]))
            

    def complete_epoch(self) -> None:
        for key in self.mean.keys():
            samples_per_dim = sum(self.batch_size[key][-1])   
            self.mean[key][-1] = (np.asarray(self.mean[key][-1])*np.asarray(self.batch_size[key][-1])).sum()/samples_per_dim
            self.std[key][-1] = (np.asarray(self.std[key][-1])*np.asarray(self.batch_size[key][-1])).sum()/samples_per_dim
            self.max[key][-1] = np.max(self.max[key][-1])
            self.min[key][-1] = np.max(self.min[key][-1])
            self.freq[key][-1] = sum([sum(freq) for freq in self.freq[key][-1]])/samples_per_dim
            
            



class FlexPPO(FlexOnPolicyMixin, PPO):
    def __init__(
        self,
        policy: Union[str, Type[FlexActorCritic]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
    
        # call OnPolicyAlgorithm initializer
        super(PPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
                spaces.Tuple,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()


    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:
            self.rollout_buffer_class = FlexSizeRolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde, 
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)
        
    
        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

          
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        clipping_stats = ClippingStats(dims=np.unique(self.rollout_buffer.get_buffered_action_dims())) 
               
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            
            clipping_stats.start_epoch()
            
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                   
                # Batch data
                observations = self.policy.batch_observations(obs=rollout_data.observations)
                actions = torch.from_numpy(np.hstack(rollout_data.actions)).unsqueeze(dim=1).to(self.device)
                dim_actions = torch.LongTensor([len(a) for a in rollout_data.actions]).to(self.device)
                values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
                values =  global_mean_pool(x=values, batch=observations["batch"]).flatten()
                log_prob =  global_add_pool(x=log_prob, batch=observations["batch"]).flatten()
                entropy = global_add_pool(x=entropy, batch=observations["batch"]).flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                clipped_ratio =  torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                               
                policy_loss_1 = advantages * ratio
                # policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss_2 = advantages * clipped_ratio
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                
                # Queue clipping statistcs
                clipping_stats.queue_batch(
                    dims=dim_actions.detach().cpu().numpy(),
                    ratio=ratio.detach().cpu().numpy(),
                    clipped_ratio=clipped_ratio.detach().cpu().numpy())
                
                    
                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob/dim_actions)
                else:
                    entropy_loss = -torch.mean(entropy/dim_actions)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    #log_ratio = log_prob - old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            
            clipping_stats.complete_epoch()
            
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        # Log clipping statistics
        for dim_key in clipping_stats.keys():
            dim_clip_stat = clipping_stats[dim_key]
            for key, value in dim_clip_stat.items():
                self.logger.record("train_clip_%s/dim_%s"%(key,dim_key), value)


    def learn(
        self: SelfFlexPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "FlexPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfFlexPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )






class VarFlexPPO(FlexPPO):
    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        mean_clip_range = []

        continue_training = True
        clipping_stats = ClippingStats(dims=np.unique(self.rollout_buffer.get_buffered_action_dims())) 
 
               
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            
            clipping_stats.start_epoch()
            
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                   
                # Batch data
                observations = self.policy.batch_observations(obs=rollout_data.observations)
                actions = torch.from_numpy(np.hstack(rollout_data.actions)).unsqueeze(dim=1).to(self.device)
                dim_actions = torch.LongTensor([len(a) for a in rollout_data.actions]).to(self.device)
                values, log_prob, entropy = self.policy.evaluate_actions(observations, actions)
                values =  global_mean_pool(x=values, batch=observations["batch"]).flatten()
                log_prob =  global_add_pool(x=log_prob, batch=observations["batch"]).flatten()
                entropy = global_add_pool(x=entropy, batch=observations["batch"]).flatten()

                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                dim_clip_range = clip_range*torch.sqrt(dim_actions.float())
                clipped_ratio =  torch.clamp(ratio, 1 - dim_clip_range, 1 + dim_clip_range)
                               
                policy_loss_1 = advantages * ratio
                # policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss_2 = advantages * clipped_ratio
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > dim_clip_range).float()).item()
                clip_fractions.append(clip_fraction)
                mean_clip_range.append(dim_clip_range.mean().item())
                
                # Queue clipping statistcs
                clipping_stats.queue_batch(
                    dims=dim_actions.detach().cpu().numpy(),
                    ratio=ratio.detach().cpu().numpy(),
                    clipped_ratio=clipped_ratio.detach().cpu().numpy())
                
                    
                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob/dim_actions)
                else:
                    entropy_loss = -torch.mean(entropy/dim_actions)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            
            clipping_stats.complete_epoch()
            
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/mean_clip_range", np.mean(mean_clip_range))
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        # Log clipping statistics
        for dim_key in clipping_stats.keys():
            dim_clip_stat = clipping_stats[dim_key]
            for key, value in dim_clip_stat.items():
                self.logger.record("train_clip_%s/dim_%s"%(key,dim_key), value)
    
    def learn(
        self: SelfFlexPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "VarFlexPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfFlexPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )