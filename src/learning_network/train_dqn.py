import gym
from gym_grid_driving.envs.grid_driving import LaneSpec, SparseReward

from .dqn_trainer import DQNTrainer
from .model import DQNModel

ENV_CONFIG = {
    'lanes': [
        LaneSpec(cars=3, speed_range=[-2, -1]),
        LaneSpec(cars=4, speed_range=[-2, -1]),
        LaneSpec(cars=2, speed_range=[-1, -1]),
        LaneSpec(cars=2, speed_range=[-3, -1]),
    ],
    'width': 10,
    'random_lane_speed': False,
    'ensure_initial_solvable': False,

    'agent_speed_range': [-2, -1],

    'rewards': SparseReward,
    'observation_type': 'tensor',
    'mask': None,
    'stochasticity': 0.0,

    'random_seed': 0,
}


def load_env(env_config):
    return gym.make('GridDriving-v0', **env_config)


def load_model(input_shape, num_actions):
    return DQNModel(input_shape, num_actions, dropout_rate=0.1)


def main(env_config):
    env = load_env(env_config)
    model = load_model(env.observation_space.shape, env.action_space.n)

    trainer = DQNTrainer(
        env,
        model,
        log_dir='runs/dqn',
        save_path='models/dqn')

    trainer.train()


if __name__ == '__main__':
    main(ENV_CONFIG)
