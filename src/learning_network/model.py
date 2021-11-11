import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_model import BaseModel


class DQNModel(BaseModel):
    def __init__(
            self,
            input_shape,
            num_actions,
            dropout_rate=0.1,
            **kwargs):
        super(DQNModel, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.dropout_rate = dropout_rate

        modifier = 0
        layer_1_out = 2 << (4 + modifier)
        layer_2_out = 2 << (5 + modifier)
        layer_3_out = 2 << (6 + modifier)
        fully_connected_out = 2 << (7 + modifier)

        self.batch_norm_input = nn.BatchNorm2d(self.input_shape[0])

        self.conv_1 = nn.Conv2d(self.input_shape[0], layer_1_out, kernel_size=2, bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(layer_1_out)
        self.conv_2 = nn.Conv2d(layer_1_out, layer_2_out, kernel_size=2, bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(layer_2_out)
        self.conv_3 = nn.Conv2d(layer_2_out, layer_3_out, kernel_size=2, bias=False)
        self.batch_norm_3 = nn.BatchNorm2d(layer_3_out)

        feature_out_size = self.calc_feature_size(input_shape)
        self.feature_layer = nn.Linear(feature_out_size, fully_connected_out)
        self.action_layer = nn.Linear(fully_connected_out, self.num_actions)

        self.dropout_2d = nn.Dropout2d(self.dropout_rate)

        self.init_parameters()

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

    def config(self):
        config = {
            'input_shape': self.input_shape,
            'num_actions': self.num_actions,
            'dropout_rate': self.dropout_rate,
        }
        return config

    def init_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.conv_1.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.conv_2.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.conv_3.weight, nonlinearity='relu')

            nn.init.kaiming_uniform_(
                self.feature_layer.weight, nonlinearity='relu')
            nn.init.constant_(self.feature_layer.bias, 0)

            nn.init.kaiming_uniform_(
                self.action_layer.weight, nonlinearity='linear')
            nn.init.constant_(self.feature_layer.bias, 0)

    def reset_parameters(self):
        self.batch_norm_input.reset_parameters()

        self.conv_1.reset_parameters()
        self.batch_norm_1.reset_parameters()

        self.conv_2.reset_parameters()
        self.batch_norm_2.reset_parameters()

        self.conv_3.reset_parameters()
        self.batch_norm_3.reset_parameters()

        self.feature_layer.reset_parameters()
        self.action_layer.reset_parameters()

        self.init_parameters()

    def forward(self, features, **kwargs):
        features = self.batch_norm_input(features)

        features = self.conv_1(features)
        features = self.batch_norm_1(features)
        features = F.relu(features)
        features = self.dropout_2d(features)

        features = self.conv_2(features)
        features = self.batch_norm_2(features)
        features = F.relu(features)
        features = self.dropout_2d(features)

        features = self.conv_3(features)
        features = self.batch_norm_3(features)
        features = F.relu(features)
        features = self.dropout_2d(features)

        features = torch.flatten(features, start_dim=1, end_dim=-1)
        features = self.feature_layer(features)
        features = F.relu(features)
        features = self.action_layer(features)

        outputs = {
            'rewards': features,
        }

        return outputs
