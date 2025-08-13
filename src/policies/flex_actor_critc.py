from typing import Type, Optional, Union, List, Dict, Any, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

import torch
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_add_pool

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.utils import obs_as_tensor

from src.common.layers.features_extractor import AugmentedGraphExtractor
from src.common.utils import batch_vec_obs, batch_to_dict, pool_actions



class FlexActorCritic(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        log_std_init: float = 0.0,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = AugmentedGraphExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        mlp_extractor_class: Type[MlpExtractor] = MlpExtractor,
        mlp_extractor_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):


        super(ActorCriticPolicy, self).__init__(
            observation_space.common_space,
            action_space.common_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
            normalize_images=normalize_images,
        )

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
        
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5
        
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs

        self.mlp_extractor_class = mlp_extractor_class
        self.mlp_extractor_kwargs = mlp_extractor_kwargs
        
        # Default network architecture, from stable-baselines
        if net_arch is None:
            net_arch = dict(pi=[64, 64], vf=[64, 64])

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.share_features_extractor = share_features_extractor
        self.features_extractor = self.make_features_extractor()
        self.features_dim = self.features_extractor.features_dim
        if self.share_features_extractor:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.features_extractor
        else:
            self.pi_features_extractor = self.features_extractor
            self.vf_features_extractor = self.make_features_extractor()

        self.log_std_init = log_std_init

        # Action distribution
        self.action_dist = DiagGaussianDistribution(action_dim=np.prod(self.action_space.shape))

        self._build(lr_schedule)
        
        



    def extract_features(
            self,
            obs: Dict,
            feature_extractor: Optional[BaseFeaturesExtractor] = None,
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            """
            Preprocess the observation if needed and extract features.

            :param obs: Observation
            :param features_extractor: The features extractor to use. If None, then ``self.features_extractor`` is used.
            :return: The extracted features. If features extractor is not shared, returns a tuple with the
                features for the actor and the features for the critic.
            """
            
            if feature_extractor == None:
                if self.share_features_extractor:
                    return self.pi_features_extractor(**obs)
                else:
                    pi_features = self.pi_features_extractor(**obs)
                    vf_features = self.vf_features_extractor(**obs)
                    return pi_features, vf_features
                
            else:
                return feature_extractor(**obs)

    def _get_action_dist_from_latent(
        self, 
        latent_pi: torch.Tensor,
        batch: Optional[Batch]=None,
        ) -> Union[DiagGaussianDistribution, List[DiagGaussianDistribution]]:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        :batch: wheter differently sized distriubtion shall be returned or scalar dimensional
        """
        
        mean_actions = self.action_net(latent_pi)
        
        if batch == None:
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        else:
            mean_actions = pool_actions(actions=mean_actions, batch=batch)
            return [self.action_dist.proba_distribution(mu) for mu in mean_actions]

        

        

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )
        
    
    def unscale_action(self, scaled_action: List[np.ndarray]) -> List[np.ndarray]:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        return [super().unscale_action(action) for action in scaled_action]
        


    def forward(self, obs, deterministic = False):    
        features = self.extract_features(obs=obs)
        
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
            
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf).squeeze()
        distribution = self._get_action_dist_from_latent(latent_pi)
        
        if not isinstance(distribution, DiagGaussianDistribution):
            raise NotImplementedError("Action distribution is not implemented yet. Be cautious inheriting from \
                DiagGaussianDistribution as the computation of the logprob below requires uncorrelated actions.")
        
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        
        return actions, values, log_prob, entropy
    
    
    def batch_observations(
        self,
        obs: Union[Sequence[Dict[str, np.ndarray]], Dict[str, np.ndarray]],
    ) -> Dict[str, torch.Tensor]:
        
        if not isinstance(obs, Sequence):
            obs = [obs]
        
        obs = [obs_as_tensor(o, self.device) for o in obs]
        obs_batch = batch_vec_obs(obs, edge_batch=True)
        obs_dict = batch_to_dict(batch=obs_batch)
        
        if "global_space" in self.observation_space.keys():
            obs_dict["global_space"] = torch.stack([data["global_space"] for data in obs], dim=0)
                
        return obs_dict
        
    def get_distribution(self, obs, batch: Optional[Batch]=None):
        """
        Get the current policy distribution given the observations.

        :param obs:
        :return: the action distribution.
        """
        features = self.extract_features(obs, self.pi_features_extractor)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi, batch)
        
    
    def predict_values(self, obs: Dict) -> torch.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        
        features = self.extract_features(obs=obs)
        #features = super().extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        
        return self.value_net(latent_vf).squeeze()
    
    
    def reset_noise(self, *args, **kwargs):
        raise NotImplementedError
    

    def _predict(self, observation, deterministic = False):
        """
        Get the action according to the policy for a given observation.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        return self.get_distribution(observation).get_actions(deterministic=deterministic)
    
    def predict(
        self,
        observation,
        state = None,
        episode_start = None,
        deterministic = False,
    ):
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        #obs_tensor, vectorized_env = self.obs_to_tensor(observation)
        observation = observation if isinstance(observation, Sequence) else [observation]
        obs_dict = self.batch_observations(observation)

        with torch.no_grad():
            actions = self._predict(obs_dict, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy()

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)  # type: ignore[assignment, arg-type]
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)  # type: ignore[assignment, arg-type]

        
        actions = pool_actions(actions=actions, batch=obs_dict["batch"])
        return actions, state  # type: ignore[return-value]
