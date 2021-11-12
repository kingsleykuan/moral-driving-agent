import copy
import random
from pathlib import Path

import gym
import numpy as np
import torch
from gym.envs.registration import register
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from environment.gym_grid_driving.envs.grid_driving import (
    FeatSpec, LaneSpec, ObsSpec, Point, DenseReward)
from moral_rewards.moral_data import MoralMachineDataset
from .model import DQNModel

register(
    id='MoralGridDriving-v0',
    entry_point=(
        'environment.gym_grid_driving.envs.grid_driving:MoralGridDrivingEnv')
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_EVAL_SAMPLES = 1000
MODEL_PATH = 'models/double_dqn'
DATA_PATH = 'data/moral_data_test.npz'

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
    'observations': [
        ObsSpec(1, (2, 1)),
        ObsSpec(2, (2, 3))
    ],

    'agent_speed_range': [-2, -1],

    'rewards': DenseReward,
    'observation_type': 'tensor',
    'mask': None,
    'stochasticity': 1.0,

    'random_seed': 0,
}


def load_model(model_path):
    return DQNModel.load(model_path)


def load_data(data_path):
    return MoralMachineDataset(data_path)


def load_env(env_config):
    return gym.make('MoralGridDriving-v0', **env_config)


def main(model_path, data_path, env_config):
    model = load_model(model_path)
    model = model.eval()
    model = model.to(device)

    random_seed = 0

    ground_truths = []
    choices = []
    data = load_data(data_path)
    for features in tqdm(data):
        saved = random.randint(0, 1)

        if saved == 0:
            features_1 = features['data_saved']
            features_2 = features['data_not_saved']
        else:
            features_1 = features['data_not_saved']
            features_2 = features['data_saved']

        scenario_1 = FeatSpec(1, False, 0, *features_1)
        scenario_2 = FeatSpec(2, False, 0, *features_2)

        scenario_env_config = copy.deepcopy(env_config)
        scenario_env_config['features'] = [scenario_1, scenario_2]

        for i in range(10):
            scenario_env_config['random_seed'] = random_seed
            env = load_env(scenario_env_config)
            random_seed += 1

            choice = None
            state = env.reset()
            # env.render()
            while True:
                state = torch.tensor(state, dtype=torch.float, device=device)
                state = torch.unsqueeze(state, 0)
                with torch.no_grad():
                    action = torch.argmax(model(state)['rewards']).item()

                next_state, reward, done, info = env.step(action)
                # env.render()
                state = next_state

                scenario_1_pos = scenario_env_config['observations'][0].pos
                scenario_2_pos = scenario_env_config['observations'][1].pos

                if next_state[1, scenario_1_pos[1], scenario_1_pos[0]] == 1:
                    choice = 1
                elif next_state[1, scenario_2_pos[1], scenario_2_pos[0]] == 1:
                    choice = 0

                if done:
                    break

            if choice is not None:
                ground_truths.append(saved)
                choices.append(choice)
                break

        if len(choices) >= MAX_EVAL_SAMPLES:
            ground_truths = np.asarray(ground_truths)
            choices = np.asarray(choices)
            accuracy = accuracy_score(ground_truths, choices)

            print(accuracy)
            return accuracy


def evaluate_incrementally(model_path, data_path, env_config, num_ignore=10):
    model_path = Path(model_path)
    model_paths = model_path.parent.glob(f'**/{model_path.name}_*')
    model_paths = [
        (int(path.name.split('_')[-1]), path)
        for path in model_paths]
    model_paths.sort(key=lambda x: x[0])

    accuracies = []
    for episode, path in tqdm(model_paths[num_ignore:]):
        accuracy = main(path, data_path, env_config)
        accuracies.append((episode, accuracy))

    np.save(model_path / 'accuracy.npy', np.asarray(accuracies))


if __name__ == '__main__':
    main(MODEL_PATH, DATA_PATH, ENV_CONFIG)
    # evaluate_incrementally(MODEL_PATH, DATA_PATH, ENV_CONFIG, num_ignore=10)
