#!/usr/bin/env python3
"""
benchmark_humanoid.py — DUSDi dmc_humanoid_state skill diversity benchmark

Loads a pretrained DUSDi actor checkpoint and rolls out every skill combination.
Metrics collected (matching TIME's metrics.py conventions):
  1. (x, y) torso trajectories per skill  →  skill_trajectories.png
  2. Mean-obs embeddings, cosine-distance matrix  →  cosine_distance_matrix.png
  3. Episode returns  →  returns.csv + printed summary

Usage:
    python benchmark_humanoid.py \\
        --snapshot_dir /workspace/DUSDI_state/exp_local/.../snapshots \\
        --snapshot_ts  3000000 \\
        --skill_dim    4 \\
        --skill_channel 3 \\
        --n_episodes   5 \\
        --episode_steps 1000 \\
        --cuda_id      0 \\
        --out_dir      ./benchmark_results_humanoid

Obs layout (67 dims):
    [0:21]  joint_angles
    [21:37] spatial (head/extremities/torso_vert)
    [37:67] velocities (com + joints)
    --> skill_channel=3 partition: [0:21] | [21:37] | [37:67]

Actor input: 67 + 4*3 = 79 dims
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
from sklearn.metrics.pairwise import cosine_distances

sys.path.insert(0, str(Path(__file__).parent))
from agent.diayn_actors import Actor
import utils  # noqa: F401


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='DUSDi dmc_humanoid_state skill benchmark')
    p.add_argument('--snapshot_dir',  required=True)
    p.add_argument('--snapshot_ts',   type=int,   default=3000000)
    p.add_argument('--skill_dim',     type=int,   default=4)
    p.add_argument('--skill_channel', type=int,   default=3,
                   help='3 channels: joint_angles | spatial | velocities')
    p.add_argument('--obs_dim',       type=int,   default=67,
                   help='humanoid flat obs: 21+16+30=67')
    p.add_argument('--action_dim',    type=int,   default=21)
    p.add_argument('--hidden_dim',    type=int,   default=1024)
    p.add_argument('--n_episodes',    type=int,   default=5)
    p.add_argument('--episode_steps', type=int,   default=1000)
    p.add_argument('--cuda_id',       type=int,   default=0)
    p.add_argument('--out_dir',       type=str,   default='./benchmark_results_humanoid')
    return p.parse_args()


# ── Actor helpers ──────────────────────────────────────────────────────────────

def build_actor(obs_dim, skill_dim, skill_channel, action_dim, hidden_dim, device):
    actor_input_dim = obs_dim + skill_dim * skill_channel  # 67 + 12 = 79
    actor = Actor(
        obs_type='states',
        obs_dim=actor_input_dim,
        action_dim=action_dim,
        feature_dim=hidden_dim,
        hidden_dim=hidden_dim,
        sac=True,
        log_std_bounds=[-10, 2],
        domain='dmc_humanoid_state',
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

def get_xy(env):
    """Humanoid root has a free joint: qpos[0]=x, qpos[1]=y, qpos[2]=z."""
    qpos = env._dm_env.physics.data.qpos
    return float(qpos[0]), float(qpos[1])


def rollout(actor, env, skill_vec, episode_steps, device):
    obs = env.reset()
    obs = np.asarray(obs, dtype=np.float32).flatten()

    obs_list, xy_list = [], []
    ep_return = 0.0

    for _ in range(episode_steps):
        xy_list.append(get_xy(env))
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

    return np.array(obs_list), np.array(xy_list, dtype=np.float32), ep_return


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_trajectories(skill_xy, all_skills, skill_dim, skill_channel, out_path):
    n_skills = len(all_skills)
    cmap = plt.cm.get_cmap('hsv', n_skills)

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, skill_tuple in enumerate(all_skills):
        xy = skill_xy[skill_tuple]
        ax.plot(xy[:, 0], xy[:, 1], color=cmap(i), alpha=0.55, linewidth=0.8)
        ax.scatter(xy[0, 0], xy[0, 1], color=cmap(i), s=30, zorder=5, marker='o')

    ax.set_title(
        f'DUSDi dmc_humanoid_state skill trajectories\n'
        f'skill_dim={skill_dim}, skill_channel={skill_channel}, total={n_skills} skills'
    )
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
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

    from custom_env.dmc_gym_env import DMCGymEnv
    env = DMCGymEnv('dmc_humanoid_state', max_episode_steps=args.episode_steps)

    all_skills = list(product(range(args.skill_dim), repeat=args.skill_channel))
    n_skills = len(all_skills)
    print(f'\nBenchmarking {n_skills} skills '
          f'(skill_dim={args.skill_dim}^skill_channel={args.skill_channel}), '
          f'{args.n_episodes} episode(s) each\n')

    skill_xy      = {}   # skill_tuple -> (T_total, 2) xy array (for plotting)
    skill_xy_eps  = {}   # skill_tuple -> list of per-episode (T_ep, 2) arrays (for CSV)
    skill_embed   = {}
    skill_returns = {}

    for s_idx, skill_tuple in enumerate(all_skills):
        skill_vec = make_skill_vec(skill_tuple, args.skill_dim, args.skill_channel, device)
        ep_obs_all, ep_xy_all, ep_rets = [], [], []

        for _ in range(args.n_episodes):
            obs_seq, xy_seq, ret = rollout(actor, env, skill_vec, args.episode_steps, device)
            ep_obs_all.append(obs_seq)
            ep_xy_all.append(xy_seq)
            ep_rets.append(ret)

        skill_xy[skill_tuple]      = np.concatenate(ep_xy_all, axis=0)
        skill_xy_eps[skill_tuple]  = ep_xy_all
        skill_embed[skill_tuple]   = np.mean(np.concatenate(ep_obs_all, axis=0), axis=0)
        skill_returns[skill_tuple] = ep_rets

        xy = skill_xy[skill_tuple]
        print(f'  [{s_idx+1:3d}/{n_skills}] skill={skill_tuple}  '
              f'mean_return={np.mean(ep_rets):8.3f}  '
              f'x=[{xy[:,0].min():.1f}, {xy[:,0].max():.1f}]  '
              f'y=[{xy[:,1].min():.1f}, {xy[:,1].max():.1f}]')

    plot_trajectories(skill_xy, all_skills, args.skill_dim, args.skill_channel,
                      out_dir / 'skill_trajectories.png')

    embeddings = np.stack([skill_embed[s] for s in all_skills])
    mean_dist, dist_matrix = plot_cosine_matrix(
        embeddings, all_skills, out_dir / 'cosine_distance_matrix.png'
    )

    # ── Trajectories CSV ─────────────────────────────────────────────────────
    csv_path = out_dir / 'trajectories_dmc_humanoid_state.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['skill_tuple', 'episode', 'step', 'x', 'y'])
        for skill_tuple, eps in skill_xy_eps.items():
            for ep_idx, xy_seq in enumerate(eps):
                for step, (x, y) in enumerate(xy_seq):
                    writer.writerow([str(skill_tuple), ep_idx, step, f'{x:.4f}', f'{y:.4f}'])
    print(f'Saved: {csv_path}')

    all_returns = [r for rs in skill_returns.values() for r in rs]
    all_xy = np.concatenate(list(skill_xy.values()), axis=0)

    print('\n' + '='*55)
    print('DUSDi dmc_humanoid_state Benchmark Summary')
    print('='*55)
    print(f'  Checkpoint          : actor_{args.snapshot_ts}.pt')
    print(f'  Skills              : {n_skills} ({args.skill_dim}^{args.skill_channel})')
    print(f'  Episodes per skill  : {args.n_episodes}')
    print(f'  Episode steps       : {args.episode_steps}')
    print(f'  Mean return         : {np.mean(all_returns):.3f} ± {np.std(all_returns):.3f}')
    print(f'  Max / Min return    : {np.max(all_returns):.3f} / {np.min(all_returns):.3f}')
    print(f'  X coverage          : {all_xy[:,0].max() - all_xy[:,0].min():.2f} m')
    print(f'  Y coverage          : {all_xy[:,1].max() - all_xy[:,1].min():.2f} m')
    print(f'  Cosine dist (mean)  : {mean_dist:.4f}  (higher = more diverse skills)')
    print(f'  Cosine dist (max)   : {dist_matrix[np.triu_indices(n_skills, k=1)].max():.4f}')
    print('='*55)


if __name__ == '__main__':
    main()
