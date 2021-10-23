import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import collections
import random

from base.base_model import BaseModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory():
    def __init__(self, memory_size):
        '''
        FILL ME : This function should initialize the replay buffer `self.buffer` with maximum size of `buffer_limit` (`int`).
                  len(self.buffer) should give the current size of the buffer `self.buffer`.
        '''
        self.buffer = collections.deque([], maxlen=memory_size)
        pass

    def push(self, transition):
        '''
        FILL ME : This function should store the transition of type `Transition` to the buffer `self.buffer`.

        Input:
            * `transition` (`Transition`): tuple of a single transition (state, action, reward, next_state, done).
                                           This function might also need to handle the case  when buffer is full.
        Output:
            * None
        '''
        self.buffer.append(transition)
        pass

    def sample(self, batch_size):
        '''
        FILL ME : This function should return a set of transitions of size `batch_size` sampled from `self.buffer`

        Input:
            * `batch_size` (`int`): the size of the sample.

        Output:
            * A 5-tuple (`states`, `actions`, `rewards`, `next_states`, `dones`),
                * `states`      (`torch.tensor` [batch_size, channel, height, width])
                * `actions`     (`torch.tensor` [batch_size, 1])
                * `rewards`     (`torch.tensor` [batch_size, 1])
                * `next_states` (`torch.tensor` [batch_size, channel, height, width])
                * `dones`       (`torch.tensor` [batch_size, 1])
              All `torch.tensor` (except `actions`) should have a datatype `torch.float` and resides in torch device
              `device`.
        '''
        transitionList = random.sample(self.buffer, batch_size)
        stateTensorList = []
        actionTensorList = []
        rewardTensorList = []
        nextStateTensorList = []
        doneTensorList = []
        for transition in transitionList:
            state, action, reward, nextState, done = transition
            stateTensorList.append(state)
            actionTensorList.append(action)
            rewardTensorList.append(reward)
            nextStateTensorList.append(nextState)
            doneTensorList.append(done)

        '''
        May not need for proj. Use for testing in assignment env.
        '''
        stateTensor = torch.tensor(stateTensorList).float().to(device)
        actionTensor = torch.tensor(actionTensorList).to(device)
        rewardTensor = torch.tensor(rewardTensorList).float().to(device)
        nextStateTensor = torch.tensor(nextStateTensorList).float().to(device)
        doneTensor = torch.tensor(doneTensorList).float().to(device)

        return stateTensor, actionTensor, rewardTensor, nextStateTensor, doneTensor
        pass

    def __len__(self):
        '''
        Return the length of the replay buffer.
        '''
        return len(self.buffer)


class DQNModel(BaseModel):
    def __init__(
            self,
            input_size,
            num_actions,
            dropout_rate=0.1,
            **kwargs):
        super(DQNModel, self).__init__()

        self.input_size = input_size
        self.dropout_rate = dropout_rate
        self.input_shape = input_size
        self.num_actions = num_actions

        self.fc_1 = nn.Conv2d(self.input_shape, 32, kernel_size=2)
        self.batch_norm_1 = nn.BatchNorm2d(32)
        self.fc_2 = nn.Conv2d(32, 64, kernel_size=2)
        self.batch_norm_2 = nn.BatchNorm2d(64)

        self.feature_layer = nn.Linear(1024, 256)
        self.action_layer = nn.Linear(256, self.num_actions)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.init_parameters()

    def config(self):
        config = {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'dropout_rate': self.dropout_rate,
        }
        return config

    def init_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.fc_1.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.fc_2.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.fc_3.weight, nonlinearity='linear')
            nn.init.constant_(self.fc_3.bias, 0)

    def reset_parameters(self):
        self.batch_norm_input.reset_parameters()

        self.fc_1.reset_parameters()
        self.batch_norm_1.reset_parameters()

        self.fc_2.reset_parameters()
        self.batch_norm_2.reset_parameters()

        self.fc_3.reset_parameters()

        self.init_parameters()

    def forward(self, features, **kwargs):
        # features = self.batch_norm_input(features)
        #
        # features = self.fc_1(features)
        # features = self.batch_norm_1(features)
        # features = F.relu(features)
        # features = self.dropout(features)
        #
        # features = self.fc_2(features)
        # features = self.batch_norm_2(features)
        # features = F.relu(features)
        # features = self.dropout(features)
        #
        # features = self.fc_3(features)
        # features = torch.sigmoid(features)
        # features = torch.squeeze(features, dim=-1)

        features = self.batch_norm_input(features)

        features = self.fc_1(features)
        features = self.relu1(features)
        features = self.fc_2(features)
        features = self.relu2(features)

        features = self.feature_in_layer(features)
        features = self.relu3(features)
        features = self.action_out_layer(features)
        features = torch.sigmoid(features)
        features = torch.squeeze(features, dim=-1)

        outputs = {
            'rewards': features,
        }

        return outputs
