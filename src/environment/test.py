"""
Note: Run this from the 'src' directory using:
python -m environment.test
"""

import gym
from gym.envs.registration import register
from environment.gym_grid_driving.envs.grid_driving import LaneSpec, Point, FeatSpec, ObsSpec

### Sample test cases.
test_config = {'lanes': [LaneSpec(1, [-2, -1])] * 5,
               'width': 9,
               'gamma': 0.9,
               'seed': 15,
               'fin_pos': Point(0, 0),
               'agent_pos': Point(8, 4),
               'stochasticity': 1.0,
               'features': [
                   FeatSpec(1, True, 10, [0, 1], [0, 1], [0, 1], [0, 3], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10],
                            [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10],
                            [0, 10], [0, 10], [0, 10]),
                   FeatSpec(2, True, 10, [0, 1], [0, 1], [0, 1], [0, 3], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10],
                            [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10], [0, 10],
                            [0, 10], [0, 10], [0, 10])],
               'observations': [ObsSpec(1, (1, 1)),
                                ObsSpec(2, (1, 2))],
               }

test_case_number = 1
LANES = test_config['lanes']
WIDTH = test_config['width']
RANDOM_SEED = test_config['seed']
GAMMA = test_config['gamma']
FIN_POS = test_config['fin_pos']
AGENT_POS = test_config['agent_pos']
stochasticity = test_config['stochasticity']
features = test_config['features']
observations = test_config['observations']

register(
    id='MoralGridDriving-v0',
    entry_point=(
        'environment.gym_grid_driving.envs.grid_driving:MoralGridDrivingEnv')
)

env = gym.make('MoralGridDriving-v0', lanes=LANES, width=WIDTH,
               agent_speed_range=(-3, -1), finish_position=FIN_POS, agent_pos_init=AGENT_POS,
               stochasticity=stochasticity, features=features, observations=observations, tensor_state=False,
               flicker_rate=0., mask=None, random_seed=RANDOM_SEED)
actions = env.actions
env.reset()
env.render()
a = env.step(actions[1])
env.render()
