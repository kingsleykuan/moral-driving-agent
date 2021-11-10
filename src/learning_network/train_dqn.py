import gym
from gym.envs.registration import register

from environment.gym_grid_driving.envs.grid_driving import (
    FeatSpec, LaneSpec, ObsSpec, DenseReward)
from .dqn_trainer import DQNTrainer
from .model import DQNModel

register(
    id='MoralGridDriving-v0',
    entry_point=(
        'environment.gym_grid_driving.envs.grid_driving:MoralGridDrivingEnv')
)

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
        ObsSpec(1, (1, 1)),
        ObsSpec(2, (1, 2))
    ],

    'agent_speed_range': [-2, -1],

    'rewards': DenseReward,
    'observation_type': 'tensor',
    'mask': None,
    'stochasticity': 0.0,

    'random_seed': 0,
}


def load_env(env_config):
    return gym.make('MoralGridDriving-v0', **env_config)


def load_model(input_shape, num_actions):
    return DQNModel(input_shape, num_actions, dropout_rate=0.1)


def main(env_config):
    env = load_env(env_config)
    model = load_model(env.observation_space.shape, env.action_space.n)

    trainer = DQNTrainer(
        env,
        model,
        log_dir='runs/double_dqn',
        save_path='models/double_dqn',
        double_dqn=True)

    trainer.train()


if __name__ == '__main__':
    main(ENV_CONFIG)
