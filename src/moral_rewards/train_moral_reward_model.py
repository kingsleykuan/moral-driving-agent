#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from moral_rewards.model import MoralRewardModel
from moral_rewards.moral_data import MoralMachineDataset
from base.trainer import Trainer
from base.utils import recursive_to_device


CONFIG = {
    'data_path_train': 'data/moral_data_train.npz',
    'data_path_val': 'data/moral_data_val.npz',

    'save_path': 'models/moral_reward',
    'log_dir': 'runs/moral_reward',

    'num_epochs': 50,
    'steps_per_log': 100,
    'epochs_per_eval': 5,
    'batch_size': 8192,
    'num_workers': 4,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,

    'num_features': 23,
    'hidden_size': 128,
    'dropout_rate': 0.1,

    'random_seed': 0,
}


def load_data(
        data_path,
        batch_size=1024,
        num_workers=4,
        shuffle=True):
    dataset = MoralMachineDataset(data_path)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True)
    return data_loader


def load_model(num_features, hidden_size, dropout_rate=0.5):
    model = MoralRewardModel(
        num_features, hidden_size, dropout_rate=dropout_rate)
    return model


class MoralRewardTrainer(Trainer):
    def __init__(
            self,
            data_loader_train,
            data_loader_eval,
            model,
            num_epochs=100,
            steps_per_log=1,
            epochs_per_eval=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            weight_decay=0,
            log_dir=None,
            save_path=None):
        super().__init__(
            data_loader_train,
            data_loader_eval,
            model,
            num_epochs=num_epochs,
            steps_per_log=steps_per_log,
            epochs_per_eval=epochs_per_eval,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            log_dir=log_dir,
            save_path=save_path)
        self.current_f1_score = float('-inf')
        self.best_f1_score = float('-inf')
        self.best_epoch = None

    def train_forward(self, data):
        data = recursive_to_device(data, self.device, non_blocking=True)

        # Join data into [batch_size * 2, num_features]
        data_not_saved = data['data_not_saved']
        data_saved = data['data_saved']
        data = torch.cat((data_not_saved, data_saved), dim=0)

        rewards = self.model(data)['rewards']

        # Split rewards into not saved and saved
        rewards_not_saved, rewards_saved = torch.chunk(rewards, 2)
        outputs = {
            'rewards_not_saved': rewards_not_saved,
            'rewards_saved': rewards_saved,
        }

        # Join data into [batch_size, 2]
        # Reward correct preference (which group to sacrifice)
        rewards = torch.stack((rewards_not_saved, rewards_saved), dim=-1)
        loss = F.cross_entropy(
            rewards, torch.zeros_like(rewards_saved, dtype=int))

        return outputs, loss

    def train_log(self, outputs, loss):
        if self.writer is not None:
            self.writer.add_scalar('loss', loss, global_step=self.global_step)

    def eval_forward(self, data):
        data = recursive_to_device(data, self.device, non_blocking=True)

        # Join data into [batch_size * 2, num_features]
        data_not_saved = data['data_not_saved']
        data_saved = data['data_saved']
        data = torch.cat((data_not_saved, data_saved), dim=0)

        rewards = self.model(data)['rewards']

        # Split rewards into not saved and saved
        rewards_not_saved, rewards_saved = torch.chunk(rewards, 2)
        outputs = {
            'rewards_not_saved': rewards_not_saved,
            'rewards_saved': rewards_saved,
        }

        return outputs, None

    def eval_log(self, outputs_list, labels_list):
        rewards_not_saved = [
            outputs['rewards_not_saved'] for outputs in outputs_list]
        rewards_saved = [
            outputs['rewards_saved'] for outputs in outputs_list]

        rewards_not_saved = torch.cat(rewards_not_saved, dim=0)
        rewards_saved = torch.cat(rewards_saved, dim=0)

        # Correct preference (which group to sacrifice)
        preferences = rewards_not_saved > rewards_saved
        preferences = preferences.cpu().numpy()

        self.current_f1_score = f1_score(
            np.ones_like(preferences),
            preferences,
            average='micro',
            zero_division=0)
        accuracy = accuracy_score(np.ones_like(preferences), preferences)

        if self.current_f1_score >= self.best_f1_score:
            self.best_f1_score = self.current_f1_score
            self.best_epoch = self.epoch

        if self.writer is not None:
            self.writer.add_scalar(
                'f1_score', self.current_f1_score, global_step=self.epoch)
            self.writer.add_scalar(
                'accuracy', accuracy, global_step=self.epoch)

        print(f'F1 Score: {self.current_f1_score}')
        print(f'Accuracy: {accuracy}')

    def save_model(self):
        if self.save_path is not None and \
                self.current_f1_score >= self.best_f1_score:
            self.model.save(
                self.save_path,
                epoch=self.best_epoch,
                f1_score=self.best_f1_score)


def main(
        data_path_train,
        data_path_val,
        save_path,
        log_dir=None,
        num_epochs=50,
        steps_per_log=100,
        epochs_per_eval=5,
        batch_size=1024,
        num_workers=4,
        learning_rate=1e-3,
        weight_decay=1e-5,
        num_features=23,
        hidden_size=32,
        dropout_rate=0.5,
        random_seed=0):
    torch.manual_seed(random_seed)

    data_loader_train = load_data(
        data_path_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True)

    data_loader_eval = load_data(
        data_path_val,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False)

    model = load_model(num_features, hidden_size, dropout_rate=dropout_rate)

    trainer = MoralRewardTrainer(
        data_loader_train,
        data_loader_eval,
        model,
        num_epochs=num_epochs,
        steps_per_log=steps_per_log,
        epochs_per_eval=epochs_per_eval,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        log_dir=log_dir,
        save_path=save_path)

    trainer.train()


if __name__ == '__main__':
    main(**CONFIG)
