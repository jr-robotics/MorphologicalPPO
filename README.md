# Beyond Fixed Morphologies: Learning Graph Policies with Trust Region Compensation in Variable Action Spaces

Official code release for the paper:

> **Beyond Fixed Morphologies: Learning Graph Policies with Trust Region Compensation in Variable Action Spaces**  
> *Thomas Gallien*  


<p align="center">
  <img src="animation.gif" alt="animated gif">
</p>

---

## Abstract

Trust region methods such as **TRPO** and **PPO** are a cornerstone of modern reinforcement learning, offering stability and strong performance in continuous control. At the same time, there is growing interest in **morphological generalization** — enabling policies to operate across agents with different kinematic structures. **Graph-based policy architectures** naturally encode these structural differences, but the impact of **variable action space dimensionality** on trust region optimization remains unclear.

In this work, we present a **theoretical and empirical study** of how action space variation influences the optimization landscape under KL-divergence constraints (TRPO) and clipping penalties (PPO). We introduce a **dimension compensation**
mechanism** providing fair policy updates among varying action space dimensions.

Experiments in the **Gymnasium Swimmer** environment, where morphology can be systematically altered without changing the underlying task, show that TRC improves stability and generalization in graph-based policies.

---

## Features

- Implementation of **Graph Neural Network policies** for variable action spaces.
- **Trust Region Dimension Compenation** for stability in policy optimization.
- Support for **morphology-varying RL tasks**.
- Training and evaluation scripts for all experiments in the paper.
- Modular design for easy extension to custom environments.

---


## Installation

> **Note:**  
> • All experiments in the paper were conducted on **Ubuntu 20.04.6 LTS (CUDA 12.2)** with **Python 3.11**.  
> • All code is developed based on **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)** and we used **[Hydra](https://hydra.cc/)** for the configuration manangement.

   ```bash
   git clone https://github.com/jr-robotics/MorphologicalPPO.git
   cd MorphologicalPPO

   conda create --name venv python=3.11
   conda activate venv
   pip install -r requirements.txt
   ```


## Repository Structure

    MorphologicalPPO/
    ├── config/                 # Hydra configuration files
    │   ├── agent/              # Agent-specific settings
    │   ├── callbacks/          # Callbacks during training/evaluation
    │   ├── env/                # Environment definitions and parameters
    │   ├── hparams_search/     # Hyperparameter search configs
    │   ├── learner/            # Learning wrapper configs
    │   ├── policy/             # Policy architecture and parameters
    │   ├── env/                # Environment definitions and parameters
    │   ├── hparams_search/     # Hyperparameter search configs
    │   └── train.yaml          # Main training configuration
    ├── src/                    # Core source code
    │   ├── agents/             # Action space agnostic PPO implementation
    │   ├── common/             # Supporting code for buffers, callbacks, etc.
    │   ├── envs/               # Environment adaption based on Swimmer-v5
    │   ├── models/             # GNN-based policy networks
    │   ├── models/             # GNN-based actor-critic p
    │   ├── utils/              # Helpers for instantiation and postprocessing
    │   └── wrappers/           # Code wrappers 
    ├── eval_best.py            # Evaluates best snapshots and renders envs
    ├── inference.py            # Inference script evaluating policy snapshots
    ├── train.py                # Main training entry point
    ├── requirements.txt        # Python dependencies
    ├── README.md               # This file
    └── LICENSE                 # License information


## Usage

> All commands assume you are running from the repository root with the virtual 
> Hydra will automatically create output directories under `logs/runs` or `logs/multiruns`.
---

### 1. Single Training Run

Run training with the default `train.yaml` configuration:
```bash
python train.py
```

Trainings with alternative configurations are done via:
```bash
python train.py cfg=your_config
```

---

### 2. Hyperparameter Search
Hyperparamter serarch is done via an experiment config. For example,
 
```bash
python train.py -m experiment=hparams_var_flex_ppo
```
runs the hyperparameters search described in the paper for the PPO version compensating the action space dimensions. Other agents can easily loaded by either creating a dedictated experiment file or orverloading the agent key in bash:
```bash
python train.py -m experiment=hparams_var_flex_ppo agent=sb3_flex_ppo
```

### 3. Inference

Run 
```bash
python inference.py --run logs/multiruns/../<run_dir>
```
to evaluate all checkpoints found in <run_dir>/checkpoints/periodic and stores the mean episodic reward in <run_dir>/inference/evaluation_results.csv

Running the code below evaluates the best policy (<run_dir>/checkpoints/best_model/best_model.zip) and stores rendered images to (<run_dir>/images) along with an mp4 video.
```bash
python render_policy.py --run logs/multiruns/../<run_dir>
```
However, this requires that (<run_dir>/inference/config.yaml) exist.

To reproduce the results from the papare run:
```bash
python train.py -m experiment=multiseed_grid agent=sb3_flex_ppo,sb3_flex_varppo # vary seeds
python inference.py --run logs/multiruns/inference/<timestamp>/0/               # inference agent 1
python inference.py --run logs/multiruns/inference/<timestamp>/1/               # inference agent 2
python visualize_inference.py --run /logs/multiruns/multiseed_grid/<timestamp>  # evaluates and plots
```