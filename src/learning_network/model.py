import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import collections
import random

from base.base_model import BaseModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
transition = []
Transition = transition.fromkeys(["state", "action", "reward", "next_state", "done"])

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
        transition_dict_list = random.sample(self.buffer, batch_size)
        state_tensor_list = []
        action_tensor_list = []
        reward_tensor_list = []
        next_state_tensor_list = []
        done_tensor_list = []
        for transition in transition_dict_list:
            state_tensor_list.append(transition["state"])
            action_tensor_list.append(transition["action"])
            reward_tensor_list.append(transition["reward"])
            next_state_tensor_list.append(transition["next_state"])
            done_tensor_list.append(transition["done"])

        '''
        May not need for proj. Use for testing in assignment env.
        '''
        stateTensor = torch.tensor(state_tensor_list).float().to(device)
        actionTensor = torch.tensor(action_tensor_list).to(device)
        rewardTensor = torch.tensor(reward_tensor_list).float().to(device)
        nextStateTensor = torch.tensor(next_state_tensor_list).float().to(device)
        doneTensor = torch.tensor(done_tensor_list).float().to(device)

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
        self.num_actions = num_actions

        modifier = 0
        layer_1_out = 2 << (4 + modifier)
        layer_2_out = 2 << (5 + modifier)
        layer_3_out = 2 << (6 + modifier)
        fully_connected_out = 2 << (7 + modifier)

        self.fc_1 = nn.Conv2d(self.input_shape[2], layer_1_out, kernel_size=2, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(layer_1_out)
        self.fc_2 = nn.Conv2d(layer_1_out, layer_2_out, kernel_size=2, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(layer_2_out)
        self.fc_3 = nn.Conv2d(layer_2_out, layer_3_out, kernel_size=2, bias=False)
        self.batch_norm_3 = nn.BatchNorm2d(layer_3_out)

        feature_out_size = self.calc_feature_size(input_shape)
        self.feature_layer = nn.Linear(feature_out_size, fully_connected_out)
        self.action_layer = nn.Linear(fully_connected_out, self.num_actions)

        self.dropout = nn.Dropout(self.dropout_rate)

    def calc_feature_size(self, starting_input_shape):
        # assuming input shape following (channel, height, width)
        layers = [module for module in self.modules()]
        curr_shape = starting_input_shape
        for layer in layers:
            if (isinstance(layer, nn.Conv2d)):
                new_shape_h = int(
                    ((curr_shape[1] + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) /
                     layer.stride[0]) + 1)
                new_shape_w = int(
                    ((curr_shape[2] + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) /
                     layer.stride[1]) + 1)
                curr_shape = (layer.out_channels, new_shape_h, new_shape_w)
        numel = 1
        for dim in curr_shape:
            numel = numel * dim

        return numel

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
        features = self.fc_1(features)
        features = self.batch_norm_1(features)
        features = F.relu(features)
        features = self.dropout(features)

        features = self.fc_2(features)
        features = self.batch_norm_2(features)
        features = F.relu(features)
        features = self.dropout(features)

        features = self.fc_3(features)
        features = self.batch_norm_3(features)
        features = F.relu(features)
        features = self.dropout(features)

        features = self.feature_in_layer(features)
        features = F.relu(features)
        features = self.action_out_layer(features)

        features = torch.sigmoid(features)
        features = torch.squeeze(features, dim=-1)

        outputs = {
            'rewards': features,
        }

        return outputs
