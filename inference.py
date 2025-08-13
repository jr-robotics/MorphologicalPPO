
from argparse import ArgumentParser
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import re
import os
import time
from src.common.evaluation import evaluate_policy



def evaluate_checkpoint(snapshot_path, timestep, cfg):
    
    t0 = time.time()
    print("starting worker %d with snapshot %s..."%(os.getpid(), snapshot_path))
    
    env = hydra.utils.instantiate(cfg.env.eval_env)
    agent = hydra.utils.get_class(cfg.agent._target_).load(snapshot_path, device="cpu")

    mean_rewards, std_rewards, *_ = evaluate_policy(
        model=agent,
        env=env,
        n_eval_episodes=cfg.callbacks.eval_callback.n_eval_episodes,
        deterministic=True,
    )

    env.close()
    
    print("(pid %d, %g s) timestep: %d, mean_reward: %g, std_reward: %g"%\
        (os.getpid(), time.time()-t0, timestep, mean_rewards.mean(), std_rewards.mean()))
    
    return timestep, mean_rewards, std_rewards



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run", type=str)
    args = parser.parse_args()    
    
    run_path = Path().cwd().joinpath(args.run)
    cfg = OmegaConf.load(run_path.joinpath(".hydra", "config.yaml"))
    
    
    # fix interpolation values
    cfg.env.xml_path = run_path.joinpath("xml").as_posix()
    cfg.checkpoint_path = run_path.joinpath("inference").as_posix()
    
    # disable logging
    cfg.agent.tensorboard_log = None
     
        
    # add 15 and 20 joint env to analyze generalization
    cfg.env.eval_env.env_kwargs.append(cfg.env.eval_env.env_kwargs[-1])
    cfg.env.eval_env.env_kwargs[-1].xml_params.num_seg= 15 + 1
    cfg.env.eval_env.env_kwargs.append(cfg.env.eval_env.env_kwargs[-1])
    cfg.env.eval_env.env_kwargs[-1].xml_params.num_seg= 20 + 1
    
    # Run inference with 10 epiodes
    cfg.callbacks.eval_callback.n_eval_episodes = 10
    
    # Create inference directory if needed
    inference_dir = run_path.joinpath("inference")
    inference_dir.mkdir(parents=True, exist_ok=True)    
    
    # Results and config locations
    results_path = inference_dir.joinpath("evaluation_results.csv")
    config_path = inference_dir.joinpath("config.yaml")
    OmegaConf.save(cfg, config_path)
    
    # register environment
    hydra.utils.call(cfg.env.register)
    env = hydra.utils.instantiate(cfg.env.eval_env)   
    
    data = pd.DataFrame()
    data["steps"] = [int(re.search(r'.*_(\d+)', f.as_posix()).group(1)) for f in run_path.joinpath("checkpoints", "periodic").glob("*.zip")]
    data["snapshots"] = [f.as_posix() for f in run_path.joinpath("checkpoints", "periodic").glob("*.zip")]
    data = data.sort_values(by="steps", ascending=True).reset_index(drop=True)
    
    dofs = [c.xml_params.num_seg - 1 for c in cfg.env.eval_env.env_kwargs]

    # Preallocate DataFrame columns for all DOFs
    for dof in dofs:
        data[f"mean_reward_dim_{dof}"] = np.nan
        data[f"std_reward_dim_{dof}"] = np.nan
    
    for i, row in data.iterrows():
        
        t0 = time.time()
        timestep = row["steps"]
        snapshot = row["snapshots"]
        
        agent = hydra.utils.get_class(cfg.agent._target_).load(snapshot, device="cpu")
        mean_rewards, std_rewards, *_ = evaluate_policy(
            model=agent,
            env=env,
            n_eval_episodes=cfg.callbacks.eval_callback.n_eval_episodes,
            deterministic=True,
        )
        
        # Store results directly in DataFrame
        for env_idx, dof in enumerate(dofs):
            data.at[i, f"mean_reward_dim_{dof}"] = mean_rewards[env_idx]
            data.at[i, f"std_reward_dim_{dof}"] = std_rewards[env_idx]
            
        print("(%g s) timestep: %d, mean_reward: %g, std_reward: %g"%\
            (time.time()-t0, timestep, mean_rewards.mean(), std_rewards.mean()))

    
    data.to_csv(results_path, index=False)
    
    
    




            
        


