#!/usr/bin/env python3
"""
benchmark_cheetah.py — DUSDi dmc_cheetah_state skill diversity benchmark

Loads a pretrained DUSDi actor checkpoint and rolls out every skill combination.
Metrics collected (matching TIME's metrics.py conventions):
  1. X-position KDE density per skill (ridge/violin style)  →  skill_x_density.png
  2. Mean-obs embeddings, cosine-distance matrix            →  cosine_distance_matrix.png
  3. Episode returns                                         →  returns.csv + printed summary

Usage:
    python benchmark_cheetah.py \\
        --snapshot_dir /workspace/DUSDI_state/exp_local/.../snapshots \\
        --snapshot_ts  3000000 \\
        --skill_dim    4 \\
        --skill_channel 2 \\
        --n_episodes   1 \\
        --episode_steps 1000 \\
        --cuda_id      0 \\
        --out_dir      ./benchmark_results_cheetah
"""

import argparse
import csv
import sys
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_distances

sys.path.insert(0, str(Path(__file__).parent))
from agent.diayn_actors import Actor
import utils  # noqa: F401  — registers SquashedNormal / TruncatedNormal


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='DUSDi dmc_cheetah_state skill benchmark')
    p.add_argument('--snapshot_dir',  required=True,
                   help='Path to the snapshots/ directory from pretraining')
    p.add_argument('--snapshot_ts',   type=int,   default=3000000)
    p.add_argument('--skill_dim',     type=int,   default=4,
                   help='Number of discrete skills per channel')
    p.add_argument('--skill_channel', type=int,   default=2,
                   help='Number of skill channels (obs partition segments)')
    p.add_argument('--obs_dim',       type=int,   default=17,
                   help='Cheetah flat obs dim: 8 positions + 9 velocities')
    p.add_argument('--action_dim',    type=int,   default=6)
    p.add_argument('--hidden_dim',    type=int,   default=1024)
    p.add_argument('--n_episodes',    type=int,   default=1)
    p.add_argument('--episode_steps', type=int,   default=1000)
    p.add_argument('--cuda_id',       type=int,   default=0)
    p.add_argument('--out_dir',       type=str,   default='./benchmark_results_cheetah')
    return p.parse_args()


# ── Actor helpers ──────────────────────────────────────────────────────────────

def build_actor(obs_dim, skill_dim, skill_channel, action_dim, hidden_dim, device):
    actor_input_dim = obs_dim + skill_dim * skill_channel  # 17 + 4*2 = 25
    actor = Actor(
        obs_type='states',
        obs_dim=actor_input_dim,
        action_dim=action_dim,
        feature_dim=hidden_dim,
        hidden_dim=hidden_dim,
        sac=True,
        log_std_bounds=[-10, 2],
        domain='dmc_cheetah_state',
    )
    return actor.to(device)


def load_actor(snapshot_dir, snapshot_ts, actor, device):
    actor_path = Path(snapshot_dir) / f'actor_{snapshot_ts}.pt'
    print(f'Loading actor: {actor_path}')
    if not actor_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {actor_path}')
    state_dict = torch.load(actor_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


# ── Skill helpers ─────────────────────────────────────────────────────────────

def make_skill_vec(skill_tuple, skill_dim, skill_channel, device):
    skill = np.zeros((skill_channel, skill_dim), dtype=np.float32)
    for ch, idx in enumerate(skill_tuple):
        skill[ch, idx] = 1.0
    return torch.as_tensor(skill.flatten(), device=device)


# ── Rollout ───────────────────────────────────────────────────────────────────

def get_x_pos(env):
    """Get cheetah torso x position from dm_control physics (not in obs)."""
    # action_scale.Wrapper -> EnvironmentWrapper exposes .physics
    return float(env._dm_env.physics.named.data.qpos['rootx'])


def rollout(actor, env, skill_vec, episode_steps, device):
    """
    Run one episode with a fixed skill vector.
    Returns:
        obs_seq  : (T, obs_dim) float32
        x_seq    : (T,) float32  — torso x position
        ep_return: float
    """
    obs = env.reset()
    obs = np.asarray(obs, dtype=np.float32).flatten()

    obs_list, x_list = [], []
    ep_return = 0.0

    for _ in range(episode_steps):
        x_list.append(get_x_pos(env))
        obs_list.append(obs.copy())

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
            inpt = torch.cat([obs_t, skill_vec], dim=-1)
            action = actor(inpt, 0.2).mean.cpu().numpy()

        obs, reward, done, _ = env.step(action)
        obs = np.asarray(obs, dtype=np.float32).flatten()
        ep_return += float(reward)
        if done:
            break

    return np.array(obs_list), np.array(x_list, dtype=np.float32), ep_return


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_x_density(skill_x, all_skills, skill_dim, skill_channel, out_path, x_lim=(-10, 30)):
    """
    KDE density ridge plot — mirrors TIME's plot_x_location_metrics.
    Each skill gets a horizontal density band stacked vertically.
    """
    n_skills = len(all_skills)
    cmap = plt.cm.get_cmap('hsv', n_skills)

    fig, ax = plt.subplots(figsize=(10, max(4, n_skills * 0.5 + 2)))

    for i, skill_tuple in enumerate(all_skills):
        x = skill_x[skill_tuple]
        if len(x) < 2:
            continue
        try:
            kde = gaussian_kde(x)
            xs = np.linspace(x.min(), x.max(), 500)
            density = kde(xs)
            density = (density - density.min()) / (density.max() + 1e-8)
            y = i * 2
            ax.fill_between(xs, y + density, y - density,
                            color=cmap(i), alpha=0.5, label=f'skill {skill_tuple}')
        except Exception:
            pass

    ax.set_xlabel('X position (m)')
    ax.set_yticks([])
    ax.set_xlim(x_lim)
    ax.set_title(
        f'DUSDi dmc_cheetah_state — X position density per skill\n'
        f'skill_dim={skill_dim}, skill_channel={skill_channel}, total={n_skills} skills'
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_cosine_matrix(embeddings, all_skills, out_path):
    dist_matrix = cosine_distances(embeddings)
    n = len(all_skills)
    mean_dist = float(np.mean(dist_matrix[np.triu_indices(n, k=1)]))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dist_matrix, vmin=0, vmax=1, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Cosine distance')
    ax.set_title(f'Skill embedding cosine-distance matrix | mean = {mean_dist:.4f}')
    ax.set_xlabel('Skill index')
    ax.set_ylabel('Skill index')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')
    return mean_dist, dist_matrix


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    actor = build_actor(args.obs_dim, args.skill_dim, args.skill_channel,
                        args.action_dim, args.hidden_dim, device)
    actor = load_actor(args.snapshot_dir, args.snapshot_ts, actor, device)

    sys.path.insert(0, str(Path(__file__).parent))
    from custom_env.dmc_gym_env import DMCGymEnv
    env = DMCGymEnv('dmc_cheetah_state', max_episode_steps=args.episode_steps)

    all_skills = list(product(range(args.skill_dim), repeat=args.skill_channel))
    n_skills = len(all_skills)
    print(f'\nBenchmarking {n_skills} skills '
          f'(skill_dim={args.skill_dim}^skill_channel={args.skill_channel}), '
          f'{args.n_episodes} episode(s) each\n')

    skill_x      = {}
    skill_embed  = {}
    skill_returns = {}

    for s_idx, skill_tuple in enumerate(all_skills):
        skill_vec = make_skill_vec(skill_tuple, args.skill_dim, args.skill_channel, device)
        ep_obs_all, ep_x_all, ep_rets = [], [], []

        for _ in range(args.n_episodes):
            obs_seq, x_seq, ret = rollout(actor, env, skill_vec, args.episode_steps, device)
            ep_obs_all.append(obs_seq)
            ep_x_all.append(x_seq)
            ep_rets.append(ret)

        skill_x[skill_tuple]      = np.concatenate(ep_x_all)
        skill_embed[skill_tuple]  = np.mean(np.concatenate(ep_obs_all, axis=0), axis=0)
        skill_returns[skill_tuple] = ep_rets

        x_arr = skill_x[skill_tuple]
        print(f'  [{s_idx+1:2d}/{n_skills}] skill={skill_tuple}  '
              f'mean_return={np.mean(ep_rets):8.3f}  '
              f'x=[{x_arr.min():.1f}, {x_arr.max():.1f}]')

    # auto-range x axis from data
    all_x = np.concatenate(list(skill_x.values()))
    x_lim = (float(all_x.min()) - 1.0, float(all_x.max()) + 1.0)

    plot_x_density(skill_x, all_skills, args.skill_dim, args.skill_channel,
                   out_dir / 'skill_x_density.png', x_lim=x_lim)

    embeddings = np.stack([skill_embed[s] for s in all_skills])
    mean_dist, dist_matrix = plot_cosine_matrix(
        embeddings, all_skills, out_dir / 'cosine_distance_matrix.png'
    )

    csv_path = out_dir / 'returns.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['skill_tuple', 'episode', 'return'])
        for skill_tuple, rets in skill_returns.items():
            for ep, ret in enumerate(rets):
                writer.writerow([str(skill_tuple), ep, f'{ret:.4f}'])
    print(f'Saved: {csv_path}')

    all_returns = [r for rs in skill_returns.values() for r in rs]
    x_range = all_x.max() - all_x.min()

    print('\n' + '='*55)
    print('DUSDi dmc_cheetah_state Benchmark Summary')
    print('='*55)
    print(f'  Checkpoint          : actor_{args.snapshot_ts}.pt')
    print(f'  Skills              : {n_skills} ({args.skill_dim}^{args.skill_channel})')
    print(f'  Episodes per skill  : {args.n_episodes}')
    print(f'  Episode steps       : {args.episode_steps}')
    print(f'  Mean return         : {np.mean(all_returns):.3f} ± {np.std(all_returns):.3f}')
    print(f'  Max / Min return    : {np.max(all_returns):.3f} / {np.min(all_returns):.3f}')
    print(f'  X coverage          : {x_range:.2f} m')
    print(f'  Cosine dist (mean)  : {mean_dist:.4f}  (higher = more diverse skills)')
    print(f'  Cosine dist (max)   : {dist_matrix[np.triu_indices(n_skills, k=1)].max():.4f}')
    print('='*55)


if __name__ == '__main__':
    main()
