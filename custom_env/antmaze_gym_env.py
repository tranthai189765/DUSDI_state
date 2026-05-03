import gym
from gym import spaces
import numpy as np


_TASK_MAP = {
    'antmaze_umaze':           'AntMaze_UMaze-v5',
    'antmaze_medium_play':     'AntMaze_Medium_Diverse_GR-v5',   # diverse goal + reset
    'antmaze_medium_diverse':  'AntMaze_Medium_Diverse_G-v5',    # diverse goal only
    'antmaze_large_play':      'AntMaze_Large_Diverse_GR-v5',    # diverse goal + reset
    'antmaze_large_diverse':   'AntMaze_Large_Diverse_G-v5',     # diverse goal only
}


class AntMazeGymEnv(gym.Env):
    """Gym wrapper for gymnasium-robotics AntMaze environments.

    Flat obs layout (gymnasium-robotics default, cfrc_ext ON):
        observation  (105) — same format as Ant-v5: [0:13] pose | [13:105] velocity+cfrc
        achieved_goal  (2) — current (x, y) position
        desired_goal   (2) — target  (x, y) position
        total          109 dims

    Action space is already [-1, 1] in AntMaze-v5.
    Returns old gym API (4-tuple step, obs-only reset) for Gym2DMWrapper.
    """

    def __init__(self, domain_key, seed=0):
        try:
            import gymnasium
            import gymnasium_robotics  # noqa: F401 — registers AntMaze envs
        except ImportError as e:
            raise ImportError(
                'gymnasium-robotics is required for AntMaze environments. '
                'Install it with: pip install gymnasium-robotics'
            ) from e

        env_id = _TASK_MAP[domain_key]
        # render_mode=None avoids EGL context issues when forked in SubprocVecEnv.
        # Antmaze is in no_video_eval_list so render() is never called during training.
        self._env = gymnasium.make(env_id, render_mode=None)
        self._seed = seed
        self._seeded = False

        obs, _ = self._env.reset(seed=seed)
        self._seeded = True
        flat_obs = self._flatten_obs(obs)
        obs_dim = flat_obs.shape[0]

        act = self._env.action_space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=act.low.astype(np.float32),
            high=act.high.astype(np.float32),
            dtype=np.float32,
        )

    def _flatten_obs(self, obs):
        parts = []
        for key in ['observation', 'achieved_goal', 'desired_goal']:
            if key in obs:
                parts.append(np.asarray(obs[key], dtype=np.float32).flatten())
        return np.concatenate(parts)

    def reset(self):
        if not self._seeded:
            obs, _ = self._env.reset(seed=self._seed)
            self._seeded = True
        else:
            obs, _ = self._env.reset()
        return self._flatten_obs(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        # Pass terminated so Gym2DMWrapper can set discount=0 on true termination
        # (goal reached), vs discount=1 on truncation (time limit) — matches TIME.
        if not isinstance(info, dict):
            info = {}
        info['terminated'] = terminated
        return self._flatten_obs(obs), float(reward), done, info

    def get_additional_states(self):
        return np.array([], dtype=np.float32)

    def seed(self, seed=None):
        pass

    def render(self, mode='rgb_array'):
        return self._env.render()
