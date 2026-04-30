## DUSDi: Disentangled Unsupervised Skill Discovery for Efficient Hierarchical Reinforcement Learning (NeurIPS 2024)

This codebase was modified based on [URLB](https://github.com/rll-research/url_benchmark).

---

## Installation

### 1. Create conda environment

```sh
conda env create -f conda_env.yml
conda activate dusdi
```

### 2. Install PyTorch with CUDA support

```sh
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu117
```

> Requires CUDA 11.7. For other CUDA versions see [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Install PettingZoo (for particle environment)

```sh
git clone https://github.com/JiahengHu/Pettingzoo-skill.git
cd Pettingzoo-skill
pip install -e .
cd ..
```

---

## Environments

### Particle (default)

No additional setup required.

### DMC — Humanoid, Quadruped & Hopper

Uses `dm_control` with MuJoCo physics. Set the rendering backend before running:

```sh
# Linux (headless server)
export MUJOCO_GL=egl

# Linux (with display)
export MUJOCO_GL=glfw

# macOS
export MUJOCO_GL=glfw
```

On headless servers (no display), `egl` requires the EGL libraries:
```sh
sudo apt-get install libgl1-mesa-dev libegl1-mesa-dev
```

---

## Pre-training

### Particle environment

```sh
python pretrain.py agent=dusdi_diayn domain=particle agent.skill_dim=5 env.particle.N=10 exp_nm="test" use_wandb=false use_tb=true
```

### DMC Humanoid (standard dm_control)

```sh
python pretrain.py domain=dmc_humanoid_state use_wandb=false use_tb=true n_env=1
```

### DMC Quadruped (standard dm_control, consistent with url_benchmark)

```sh
python pretrain.py domain=dmc_quadruped_state use_wandb=false use_tb=true n_env=1
```

### DMC Hopper (standard dm_control, task: `hopper hop`)

```sh
python pretrain.py domain=dmc_hopper_state use_wandb=false use_tb=true n_env=1
```

Algorithm hyperparameters used per DMC environment (defined in `agent/dusdi_diayn.yaml`):

| Parameter | Humanoid | Quadruped | Hopper |
|-----------|----------|-----------|--------|
| `skill_dim` | 2 | 4 | 2 |
| `update_skill_every_step` | 200 | 200 | 200 |
| `init_temperature` | 0.1 | 0.1 | 0.1 |
| `nstep` | 1 | 1 | 1 |
| `critic_type` | mask_unwt | mask_unwt | mask_unwt |
| `step_count_threshold` | 20 | 20 | 20 |
| `sac` | true | true | true |

Snapshots are saved to:
```
./models/<obs_type>/<domain>/<experiment>/<seed>/
```

Each checkpoint saves:
- `actor_<frame>.pt` — actor weights
- `critic_<frame>.pt` — critic weights
- `discriminator_<frame>.pt` — DIAYN discriminator weights
- `snapshot_<frame>.pt` — full agent object (for inference)
- `cfg_<frame>.yaml` — full config (architecture and hyperparameters)

---

## Downstream Hierarchical Learning

```sh
python train.py domain=particle ds_task=poison_l low_path="seed:2 particle dusdi_diayn test"
```

---

## Monitoring

### TensorBoard

```sh
tensorboard --logdir exp_local
```

### CSV logs

`train.csv` and `eval.csv` are written to the run directory under `exp_local/`.

### Console output

```
| train | F: 6000 | S: 3000 | E: 6 | L: 1000 | R: 5.5177 | FPS: 96.7586 | T: 0:00:42
```

| Symbol | Meaning |
|--------|---------|
| F | Total environment frames |
| S | Total agent steps |
| E | Total episodes |
| L | Episode length |
| R | Episode return |
| FPS | Training throughput |
| T | Total training time |

---

## Key changes vs. original `conda_env.yml`

- Removed `gymnasium` (conflicts with `gym==0.21.0` used by stable-baselines3)
- Removed `functorch` (built into PyTorch 2.0)
- Replaced `tb-nightly` with stable `tensorboard`
- Fixed `sklearn==0.0` → `scikit-learn`
- Pinned `dm_control==1.0.14` for reproducibility
