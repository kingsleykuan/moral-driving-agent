import gym
from gym.envs.registration import register

from environment.gym_grid_driving.envs.grid_driving import (
    FeatSpec, LaneSpec, ObsSpec, Point, DenseReward)
from .dqn_trainer import DQNTrainer
from .model import DQNModel

register(
    id='MoralGridDriving-v0',
    entry_point=(
        'environment.gym_grid_driving.envs.grid_driving:MoralGridDrivingEnv')
)

ENV_CONFIG = {
    'lanes': [
        LaneSpec(cars=1, speed_range=[-2, -1]),
        LaneSpec(cars=1, speed_range=[-2, -1]),
        LaneSpec(cars=1, speed_range=[-2, -1]),
        LaneSpec(cars=1, speed_range=[-2, -1]),
        LaneSpec(cars=1, speed_range=[-2, -1]),
    ],
    'width': 10,
    'agent_pos_init': Point(9, 2),
    'finish_position': Point(0, 2),
    'random_lane_speed': False,
    'ensure_initial_solvable': False,

    'moral_reward_model_path': 'models/moral_reward',
    'features': [
        FeatSpec(
            1, True, 5,  # id, random, max
            [0, 1],    # PedPed
            [0, 1],    # Barrier
            [0, 2],    # CrossingSignal
            [0, 5],    # Man
            [0, 5],    # Woman
            [0, 5],    # Pregnant
            [0, 5],    # Stroller
            [0, 5],    # OldMan
            [0, 5],    # OldWoman
            [0, 5],    # Boy
            [0, 5],    # Girl
            [0, 5],    # Homeless
            [0, 5],    # LargeWoman
            [0, 5],    # LargeMan
            [0, 5],    # Criminal
            [0, 5],    # MaleExecutive
            [0, 5],    # FemaleExecutive
            [0, 5],    # FemaleAthlete
            [0, 5],    # MaleAthlete
            [0, 5],    # FemaleDoctor
            [0, 5],    # MaleDoctor
            [0, 5],    # Dog
            [0, 5]),   # Cat
        FeatSpec(
            2, True, 5,  # id, random, max
            [0, 1],    # PedPed
            [0, 1],    # Barrier
            [0, 2],    # CrossingSignal
            [0, 5],    # Man
            [0, 5],    # Woman
            [0, 5],    # Pregnant
            [0, 5],    # Stroller
            [0, 5],    # OldMan
            [0, 5],    # OldWoman
            [0, 5],    # Boy
            [0, 5],    # Girl
            [0, 5],    # Homeless
            [0, 5],    # LargeWoman
            [0, 5],    # LargeMan
            [0, 5],    # Criminal
            [0, 5],    # MaleExecutive
            [0, 5],    # FemaleExecutive
            [0, 5],    # FemaleAthlete
            [0, 5],    # MaleAthlete
            [0, 5],    # FemaleDoctor
            [0, 5],    # MaleDoctor
            [0, 5],    # Dog
            [0, 5]),   # Cat
        ],
    'observations': [
        ObsSpec(1, (2, 1)),
        ObsSpec(2, (2, 3))
    ],

    'agent_speed_range': [-2, -1],

    'rewards': DenseReward,
    'observation_type': 'tensor',
    'mask': None,
    'stochasticity': 1.0,

    'random_seed': None,
}


def load_env(env_config):
    return gym.make('MoralGridDriving-v0', **env_config)


def load_model(input_channels, num_actions):
    return DQNModel(input_channels, num_actions, dropout_rate=0.1)


def load_pretrained_model(model_path):
    return DQNModel.load(model_path)


def train_base(env_config):
    env_config['moral_reward_model_path'] = None
    env = load_env(env_config)

    model = load_model(env.observation_space.shape[0], env.action_space.n)

    trainer = DQNTrainer(
        env,
        model,
        log_dir='runs/double_dqn_base',
        save_path='models/double_dqn_base',
        save_incrementally=False,
        double_dqn=True,
        double_replay=False)

    trainer.train()


def train_moral(env_config):
    env_config['moral_reward_model_path'] = 'models/moral_reward'
    env = load_env(env_config)

    model = load_pretrained_model('models/double_dqn_base')

    trainer = DQNTrainer(
        env,
        model,
        log_dir='runs/double_dqn_moral',
        save_path='models/double_dqn_moral',
        save_incrementally=True,
        double_dqn=True,
        double_replay=True)

    trainer.train()


if __name__ == '__main__':
    train_base(ENV_CONFIG)
    train_moral(ENV_CONFIG)
