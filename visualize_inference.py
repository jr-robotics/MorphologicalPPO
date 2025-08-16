from typing import Tuple, List
from argparse import ArgumentParser
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
import re
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from dataclasses import dataclass
pd.options.plotting.backend = 'plotly'

from src.utils.postprocessing import get_tensorboard_record, resolve_tags, get_synced_traces, get_async_traces


@dataclass
class PlotConfig:
    color_ppo: str
    color_varppo: str
    figure_width: int
    aspect_ratio: float
    label_font_size: int
    
    
    


    

def mm_to_px(mm: float) -> float:
    return (mm * 96 / 25.)


def hex_to_rgba(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha])

def dash_from_fraction(frac, total_length=10, min_frac=0.5):
    """
    Create a dash string where 'frac' controls the proportion
    of the dash (solid part) vs gap.
    total_length is in pixels.
    """
    
    frac = frac*(1-min_frac) + min_frac
    dash_len = int(total_length * frac)
    gap_len = total_length - dash_len
    # If frac == 1 → make it solid
    if np.isclose(frac, 1.0):
        return 'solid'
    
    # Avoid zero-length dash (would be invisible)
    dash_len = max(dash_len, 1)
    gap_len = max(gap_len, 1)
    return f"{dash_len}px,{gap_len}px"

def save_image(
    fig: go.Figure,
    filename: str,
    width_mm: float,
    aspect_ratio: float,
    **kwargs,
    ):
    
    mm_to_px = lambda mm: mm * 96 / 25.4
    height_mm = width_mm/aspect_ratio
    
    fig.update_layout(
        margin=dict(l=0.0, r=0.0, t=0.0, b=0.0),
        font=dict(size=8, family="Times New Roman", color="black"),
    )
    
    pio.write_image(
        fig=fig,
        file=filename,
        width=int(mm_to_px(width_mm)),
        height=int(mm_to_px(height_mm)),
        **kwargs,
    )
     

def split_files(multirun: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
   
    df = pd.DataFrame()#columns=["seed","agent","files"])
    runs = sorted([p for p in multirun.glob('[0-9]*')], key=lambda x: (int(x.name)))

    for run in runs:
        cfg = OmegaConf.load(run.joinpath(".hydra", "config.yaml"))
        seed = cfg.seed
        agent = cfg.agent._target_.split(".")[-1]
        
        df = pd.concat(
            [df, pd.DataFrame(dict(seed=[seed], agent=[agent], run=[run]))],
            ignore_index=True
        )
        
    # split into one dataframe per agent
    agents = sorted(df["agent"].unique())
    if len(agents) != 2:
        raise ValueError(f"Expected 2 agents, but got {len(agents)}: {agents}")

    df1 = df[df["agent"] == agents[0]].reset_index(drop=True)
    df2 = df[df["agent"] == agents[1]].reset_index(drop=True)

    # check that seeds match
    if not df1["seed"].equals(df2["seed"]):
        raise ValueError("Seeds do not match between agents!")

    return df



def plot_validation(run_path: Path, train_dims: List[int], color: str) -> go.Figure:

    df = pd.read_csv(run_path.joinpath("inference", "evaluation_results.csv"))
    val_tags = resolve_tags(obj=df, pattern="mean_reward_dim_\d+")
    val_dims = [int(re.search(r'.*_(\d+)', tag).group(1)) for tag in val_tags]

    val_tags = [tag for tag, dim in zip(val_tags, val_dims) if dim in train_dims or dim==max(val_dims)]
    val_dims = [int(re.search(r'.*_(\d+)', tag).group(1)) for tag in val_tags]

    fig = go.Figure()

    for tag, dim in zip(val_tags, val_dims):
        
        min_line_width = 1
        max_line_width = 3
        
        fraction = np.clip((dim - min(dims))/(max(dims)-min(dims)), 0 , 1)
        line_width = (max_line_width - min_line_width)*fraction + min_line_width

        dash = "solid"
        alpha = 0.5
        
        if dim not in dims:
            name = "$\dim(\mathcal{A}) = 20$"
            dash = "dash"
            alpha = 1
        elif dim==6:
            name = "$2 \leq \dim(\mathcal{A}) \leq 10$"
        else:
            name = None
        
        fig.add_trace(
            go.Scatter(
                x=df["steps"],
                y=df[tag],
                line=dict(
                    color="rgba"+str(hex_to_rgba(color, alpha)),
                    width=line_width,
                    dash=dash,
                ),
                name=name,
                showlegend=name is not None,
            )
        )
        
        

    fig.update_layout(
        xaxis=dict(title=dict(text="Timesteps"), tickformat="~s"),
        yaxis=dict(title=dict(text="Mean Episode Reward")),
            legend=dict(
            orientation='v',
            yanchor='bottom',
            y=0,
            xanchor='right',
            x=1,
            bordercolor='Black',
            borderwidth=1
        ),
    )
    
    return fig
    



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--multirun", type=str)
    parser.add_argument("--destination", type=str, default="images")
    args = parser.parse_args()    
    
    destination = Path(args.destination)
    destination.mkdir(parents=True, exist_ok=True)   
    
    plot_config = PlotConfig(
        color_ppo="#F55C5C",
        color_varppo="#001068",
        figure_width=86,
        aspect_ratio=4/3,
        label_font_size=8,
    )
    
    
    multirun = Path().cwd().joinpath(args.multirun)
    df = split_files(multirun)
    df_ppo = df[df["agent"] == "FlexPPO"].reset_index(drop=True)
    df_varppo = df[df["agent"] == "VarFlexPPO"].reset_index(drop=True)
    
    ea_varppo = get_tensorboard_record(df_varppo.iloc[0]["run"])
    ea_ppo = get_tensorboard_record(df_ppo.iloc[0]["run"])
    clip_tags = resolve_tags(obj=ea_varppo, pattern="train_clip_freq/dim_\d+")
    dims = [int(re.search(r'.*_(\d+)', tag).group(1)) for tag in clip_tags]

    
    ### Plot validation results
    
    fig_val_varppo = plot_validation(
        run_path=df_varppo.iloc[0]["run"],
        train_dims=dims,
        color=plot_config.color_varppo,
    )
    
    save_image(
        fig=fig_val_varppo,
        filename=destination.joinpath("fig_valvarppo.svg"),
        width_mm=plot_config.figure_width,
        aspect_ratio=plot_config.aspect_ratio
    )

    fig_val_ppo = plot_validation(
        run_path=df_ppo.iloc[0]["run"],
        train_dims=dims,
        color=plot_config.color_ppo,
    )
    
    save_image(
        fig=fig_val_varppo,
        filename=destination.joinpath("fig_valppo.svg"),
        width_mm=plot_config.figure_width,
        aspect_ratio=plot_config.aspect_ratio
    )
    
    
    
    ### Plot clipping fractions

    clipped_data_varppo = get_async_traces(
        ea=get_tensorboard_record(df_varppo.iloc[0]["run"]),
        tags=clip_tags
    )
    
    clipped_data_ppo = get_async_traces(
        ea=get_tensorboard_record(df_ppo.iloc[0]["run"]),
        tags=clip_tags,
    )
    
    fig_clip = go.Figure()

    for trace_varppo, trace_ppo, tag, dim in zip(clipped_data_varppo, clipped_data_ppo, clip_tags, dims):
        
        min_line_width = 1
        max_line_width = 3
        
        alpha = 1.0 if dim in [min(dims), max(dims)] else 0.5
        fraction = (dim - min(dims))/(max(dims)-min(dims))
        line_width = (max_line_width - min_line_width)*fraction + min_line_width
        
        fig_clip.add_traces(
            go.Scatter(
                x=trace_varppo["steps"],
                y=trace_varppo[tag],
                line=dict(
                    color="rgba"+str(hex_to_rgba(plot_config.color_varppo, alpha)),
                    width=line_width,
                ),
                name="$\epsilon\sqrt{\dim(\mathcal{A})}$" if dim == max(dims) else None,
                showlegend=(dim==max(dims)),
            )
        )

        fig_clip.add_traces(
            go.Scatter(
                x=trace_ppo["steps"],
                y=trace_ppo[tag],
                line=dict(
                    color="rgba"+str(hex_to_rgba(plot_config.color_ppo, alpha)),
                    width=line_width,
                ),
                name="$\epsilon$" if dim == max(dims) else None,
                showlegend=(dim==max(dims)),
            )
        )

    fig_clip.update_layout(
        xaxis=dict(title=dict(text="Timesteps"), tickformat="~s"),
        yaxis=dict(title=dict(text="Clipping Fraction")),
        legend=dict(
            orientation='v',
            yanchor='bottom',
            y=0,
            xanchor='right',
            x=1,
            bordercolor='Black',
            borderwidth=1
        ),
    )

    save_image(
        fig=fig_clip,
        filename=destination.joinpath("fig_clip.svg"),
        width_mm=plot_config.figure_width,
        aspect_ratio=plot_config.aspect_ratio
    )
    
    
    rollouts_varppo = []
    rollouts_ppo = []
    df_varppo_train = pd.DataFrame()
    df_ppo_train = pd.DataFrame()
    
    fig_train = go.Figure()
    tag_train = "rollout/ep_rew_mean"
    
    for varppo, ppo in zip(df_varppo.iterrows(), df_ppo.iterrows()):

        rollouts_varppo.append(
            get_synced_traces(
                ea=get_tensorboard_record(varppo[1].run),
                tags=tag_train,
            )
        )
        
        rollouts_ppo.append(
            get_synced_traces(
                ea=get_tensorboard_record(ppo[1].run),
                tags=tag_train,
            )
        )
        
        # outlier removal
        if rollouts_varppo[-1][tag_train].max() < 300.0 or rollouts_ppo[-1][tag_train].max() < 300.0:
            continue
        
        if "steps" not in df_varppo_train.columns:
            df_varppo_train["steps"] = rollouts_varppo[-1]["steps"]
           
        df_varppo_train["reward_%d"%varppo[1].seed] = rollouts_varppo[-1][tag_train] 
        
         
        if "steps" not in df_ppo_train.columns:
            df_ppo_train["steps"] = rollouts_ppo[-1]["steps"]
            
        df_ppo_train["reward_%d"%ppo[1].seed] = rollouts_ppo[-1][tag_train] 

        
        
        fig_train.add_trace(
            go.Scatter(
                x=rollouts_varppo[-1]["steps"],
                y=rollouts_varppo[-1]["rollout/ep_rew_mean"],
                line=dict(
                    color="rgba"+str(hex_to_rgba(plot_config.color_varppo, 0.5)),
                    width=min_line_width,
                    dash="solid",
                ),
                name=None,
                showlegend=False,
            )
        )
    
        fig_train.add_trace(
            go.Scatter(
                x=rollouts_ppo[-1]["steps"],
                y=rollouts_ppo[-1]["rollout/ep_rew_mean"],
                line=dict(
                    color="rgba"+str(hex_to_rgba(plot_config.color_ppo, 0.5)),
                    width=min_line_width,
                    dash="solid",
                ),
                name=None,
                showlegend=False,
            )
        )
        
        
    # mean line
    fig_train.add_trace(
        go.Scatter(
            x=df_varppo_train["steps"],
            y=df_varppo_train.drop(columns="steps").mean(axis=1),
            line=dict(
                color="rgba"+str(hex_to_rgba(plot_config.color_varppo, 1.0)),
                width=max_line_width,
                dash="solid",
            ),
            name="$\epsilon\sqrt{\dim(\mathcal{A})}$",
        )
    )
    
    fig_train.add_trace(
        go.Scatter(
            x=df_ppo_train["steps"],
            y=df_ppo_train.drop(columns="steps").mean(axis=1),
            line=dict(
                color="rgba"+str(hex_to_rgba(plot_config.color_ppo, 1.0)),
                width=max_line_width,
                dash="solid",
            ),
            name="$\epsilon$",
            
        )
    )
    
    fig_train.update_layout(
        xaxis=dict(title=dict(text="Timesteps")),
        yaxis=dict(title=dict(text="Mean Episode Reward")),
        legend=dict(
            orientation='v',
            yanchor='bottom',
            y=0,
            xanchor='right',
            x=1,
            bordercolor='Black',
            borderwidth=1
        ),
    )

    save_image(
        fig=fig_train,
        filename=destination.joinpath("fig_mean_reward.svg"),
        width_mm=plot_config.figure_width,
        aspect_ratio=plot_config.aspect_ratio
    )