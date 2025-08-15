from typing import Sequence 
from argparse import ArgumentParser
from pathlib import Path
import hydra
from omegaconf import OmegaConf
import numpy as np
import cv2
import pandas as pd
from src.common.evaluation import evaluate_policy


def create_mp4(frames: Sequence[np.ndarray], filepath: Path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    
    # Get the dimensions of the first image
    height, width, _ = frames[0].shape
    video = cv2.VideoWriter(filepath.as_posix(), fourcc, fps=30, frameSize=(width, height))
    
    # Write each image to the video
    for frame in frames:
        # Ensure the image is in the correct format (BGR)
        bgr_image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video.write(bgr_image)
        
    # Release the VideoWriter object
    video.release()




def render_policy(env, policy):
    frames = [[] for _ in env.envs]
    obs = env.reset()
    while True:
        action, _ = policy.predict(observation=obs, state=None, episode_start=None, deterministic=True)
        obs, _, done, _ = env.step(action)

        for n_env in range(len(env.envs)-1):
            frames[n_env].append(env.envs[n_env].render())

        if any(done):
            break
        
    return frames
    
    
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run", type=str)
    args = parser.parse_args()    
    
    run_path = Path().cwd().joinpath(args.run)
    cfg = OmegaConf.load(run_path.joinpath("inference", "config.yaml"))
    
    for env_kwargs in cfg.env.eval_env.env_kwargs:
        if "render_mode" not in env_kwargs:
            env_kwargs.update(dict(render_mode="rgb_array"))
        else:
            env_kwargs.render_mode = "rgb_array"  
    
    dofs = [c.xml_params.num_seg - 1 for c in cfg.env.eval_env.env_kwargs]
    
    
    # Render trajectories with best performing policy
    hydra.utils.call(cfg.env.register)
    env = hydra.utils.instantiate(cfg.env.eval_env)    
    best_model_path = run_path.joinpath("checkpoints", "best_model", "best_model.zip").as_posix()
    best_agent = hydra.utils.get_class(cfg.agent._target_).load(best_model_path, device="cpu")
    frames = render_policy(env, best_agent.policy)
    
    for dof, dof_frames in zip(dofs, frames):
        
        # Store frames to disc
        image_path = run_path.joinpath("images", "dim_%d"%dof)
        image_path.mkdir(parents=True, exist_ok=True)
        for frame_count, frame in enumerate(dof_frames):
            filename = image_path.joinpath(cfg.env.id + "_%d.png"%frame_count)
            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename=filename.as_posix(), img=cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
        # Create video
        create_mp4(
            frames=np.asarray(dof_frames),
            filepath=image_path.joinpath("dim_%d.mp4"%dof)
        )
        
    # Evaluate best policy
    mean_rewards, std_rewards, *_ = evaluate_policy(
        model=best_agent,
        env=env,
        n_eval_episodes=cfg.callbacks.eval_callback.n_eval_episodes,
        deterministic=True,
    )
    
    data = pd.DataFrame()
    for env_idx, dof in enumerate(dofs):
        data[f"mean_reward_dim_{dof}"] = mean_rewards[env_idx]
        data[f"std_reward_dim_{dof}"] = std_rewards[env_idx]


    data.to_csv(run_path.joinpath("inference", "best_model_evaluation.csv"), index=False)
 