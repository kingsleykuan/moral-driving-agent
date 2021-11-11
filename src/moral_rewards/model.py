import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_model import BaseModel


class MoralRewardModel(BaseModel):
    def __init__(
            self,
            input_size,
            hidden_size,
            dropout_rate=0.1,
            **kwargs):
        super(MoralRewardModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        self.batch_norm_input = nn.BatchNorm1d(self.input_size)

        self.fc_1 = nn.Linear(self.input_size, self.hidden_size, bias=False)
        self.batch_norm_1 = nn.BatchNorm1d(self.hidden_size)

        self.fc_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.batch_norm_2 = nn.BatchNorm1d(self.hidden_size)

        self.fc_3 = nn.Linear(self.hidden_size, 1)

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
        features = self.batch_norm_input(features)

        features = self.fc_1(features)
        features = self.batch_norm_1(features)
        features = F.relu(features)
        features = self.dropout(features)

        features = self.fc_2(features)
        features = self.batch_norm_2(features)
        features = F.relu(features)
        features = self.dropout(features)

        features = self.fc_3(features)
        features = -torch.sigmoid(features)
        features = torch.squeeze(features, dim=-1)

        outputs = {
            'rewards': features,
        }

        return outputs
