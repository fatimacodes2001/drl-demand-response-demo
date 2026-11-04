from .alphazero import AlphaZeroAgent
from .ddpg import DDPGAgent
from .dqn import DQNAgent
from .ppo import PPOAgent

AGENTS = {"dqn": DQNAgent, "ppo": PPOAgent, "alphazero": AlphaZeroAgent, "ddpg": DDPGAgent}
