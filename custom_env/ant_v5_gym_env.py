import gym
from gym import spaces
import numpy as np


class AntV5GymEnv(gym.Env):
    """Gym wrapper for gymnasium Ant-v5.

    Default observation (27 dims, exclude_current_positions_from_observation=True):
        [0]     z-position of torso
        [1:5]   quaternion orientation (w, x, y, z)
        [5:13]  8 joint angles (hip1, ankle1, hip2, ankle2, hip3, ankle3, hip4, ankle4)
        [13:19] 6 body velocities (vx, vy, vz, roll, pitch, yaw)
        [19:27] 8 joint velocities

    DUSDi partition used:
        channel 0 → [0:13]  body config (pose + joint angles)
        channel 1 → [13:27] velocities (body + joint)

    Action space: 8 dims, already in [-1, 1] for Ant-v5.
    Returns old gym API (4-tuple step, obs-only reset) for Gym2DMWrapper.
    """

    def __init__(self, seed=0, max_episode_steps=1000):
        try:
            import gymnasium
        except ImportError as e:
            raise ImportError(
                'gymnasium is required for Ant-v5. '
                'Install it with: pip install "gymnasium[mujoco]"'
            ) from e

        self._env = gymnasium.make(
            'Ant-v5',
            render_mode=None,
            max_episode_steps=max_episode_steps,
        )
        self._seed = seed
        self._seeded = False

        obs, _ = self._env.reset(seed=seed)
        self._seeded = True
        obs = np.asarray(obs, dtype=np.float32).flatten()
        obs_dim = obs.shape[0]

        act = self._env.action_space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=act.low.astype(np.float32),
            high=act.high.astype(np.float32),
            dtype=np.float32,
        )

    def reset(self):
        if not self._seeded:
            obs, _ = self._env.reset(seed=self._seed)
            self._seeded = True
        else:
            obs, _ = self._env.reset()
        return np.asarray(obs, dtype=np.float32).flatten()

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        if not isinstance(info, dict):
            info = {}
        info['terminated'] = terminated
        return np.asarray(obs, dtype=np.float32).flatten(), float(reward), done, info

    def seed(self, seed=None):
        pass

    def render(self, mode='rgb_array'):
        return self._env.render()
