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

        modifier = 4
        layer_1_out = 2 << (6 + modifier)
        layer_2_out = 2 << (6 + modifier)
        layer_3_out = 2 << (6 + modifier)
        fully_connected_out = 2 << (7 + modifier)

        self.batch_norm_input = nn.BatchNorm2d(self.input_shape[0])

        self.conv_1 = nn.Conv2d(self.input_shape[0], layer_1_out, kernel_size=3, padding="same", bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(layer_1_out)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv_2 = nn.Conv2d(layer_1_out, layer_2_out, kernel_size=2, padding="same", bias=False)
        self.batch_norm_2 = nn.BatchNorm2d(layer_2_out)
        self.conv_3 = nn.Conv2d(layer_2_out, layer_3_out, kernel_size=2, padding="same", bias=False)
        self.batch_norm_3 = nn.BatchNorm2d(layer_3_out)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.feature_layer = nn.Linear(layer_3_out, fully_connected_out)
        self.action_layer = nn.Linear(fully_connected_out, self.num_actions)

        self.init_parameters()

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
        features = self.max_pool(features)
        original = features

        features = self.conv_2(features)
        features = self.batch_norm_2(features)
        features = F.relu(features)

        features = self.conv_3(features)
        features = self.batch_norm_3(features)
        features = features + original
        features = F.relu(features)

        features = self.avg_pool(features)
        features = torch.flatten(features, start_dim=1, end_dim=-1)
        features = self.feature_layer(features)
        features = F.relu(features)
        features = self.action_layer(features)

        outputs = {
            'rewards': features,
        }

        return outputs
