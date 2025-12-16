import numpy as np
from gymnasium.envs.classic_control.pendulum import PendulumEnv


# Pendulum environement that always starts in neutral position
class NeutralPendulum(PendulumEnv):
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array([np.pi, 0.0], dtype=self.observation_space.dtype)
        theta, theta_dot = self.state
        return np.array([np.cos(theta), np.sin(theta), theta_dot]), {}
