from .neutral_pendulum import NeutralPendulum
from gymnasium.envs.registration import register

register(
    id="NeutralPendulum-v0",
    entry_point="mpcell.envs.neutral_pendulum:NeutralPendulum",
    max_episode_steps=200,
)
