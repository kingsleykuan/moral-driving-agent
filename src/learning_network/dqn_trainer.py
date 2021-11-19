import collections
import copy
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class ReplayMemory():
    def __init__(self, memory_size):
        self.buffer = collections.deque([], maxlen=memory_size)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        transition_dict_list = random.sample(self.buffer, batch_size)

        state_tensor_list = []
        action_tensor_list = []
        reward_tensor_list = []
        next_state_tensor_list = []
        done_tensor_list = []
        for transition in transition_dict_list:
            state_tensor_list.append(transition['state'])
            action_tensor_list.append(transition['action'])
            reward_tensor_list.append(transition['reward'])
            next_state_tensor_list.append(transition['next_state'])
            done_tensor_list.append(transition['done'])

        transitions = {
            'states': torch.stack(state_tensor_list),
            'actions': torch.stack(action_tensor_list),
            'rewards': torch.stack(reward_tensor_list),
            'next_states': torch.stack(next_state_tensor_list),
            'dones': torch.stack(done_tensor_list),
        }
        return transitions

    def __len__(self):
        return len(self.buffer)


class DQNTrainer:
    def __init__(
            self,
            env,
            model,
            num_episodes=10000,
            episode_max_len=1000,
            episodes_per_log=100,
            episodes_per_save=100,
            steps_per_log=1000,
            metrics_average_len=100,
            batch_size=64,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            weight_decay=1e-5,
            log_dir=None,
            save_path=None,
            save_incrementally=False,
            device=None,
            double_dqn=True,
            double_replay=True,
            gamma=0.999,
            epsilon_start=0.9,
            epsilon_end=0.1,
            epsilon_decay=1000,
            replay_memory_max=5000,
            replay_memory_min=1000,
            target_model_update=10):
        self.env = env
        self.model = model

        self.num_episodes = num_episodes
        self.episode_max_len = episode_max_len
        self.episodes_per_log = episodes_per_log
        self.episodes_per_save = episodes_per_save
        self.steps_per_log = steps_per_log
        self.metrics_average_len = metrics_average_len

        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.log_dir = Path(log_dir)
        self.save_path = Path(save_path)
        self.save_incrementally = save_incrementally

        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = self.model.to(self.device)
        self.model = self.model.train()

        parameter_dicts = self.model.parameter_dicts()
        self.optimizer = optim.AdamW(
            parameter_dicts,
            lr=self.learning_rate,
            weight_decay=self.weight_decay)

        if self.log_dir is not None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        else:
            self.writer = None

        self.episode = 1
        self.global_step = 1

        self.double_dqn = double_dqn
        self.double_replay = double_replay
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.replay_memory_max = replay_memory_max
        self.replay_memory_min = replay_memory_min
        self.target_model_update = target_model_update

        self.target_model = copy.deepcopy(model)
        self.target_model = self.target_model.to(self.device)
        self.target_model = self.target_model.eval()

        self.driving_replay_memory = ReplayMemory(self.replay_memory_max)
        if self.double_replay:
            self.moral_replay_memory = ReplayMemory(self.replay_memory_max)

        self.num_actions = self.env.action_space.n

        self.episode_rewards = []
        self.episode_driving_rewards = []
        self.episode_moral_rewards = []
        self.episode_finished = []

    def train(self):
        episode_rewards = []

        pbar = tqdm(
            range(self.episode, self.num_episodes + 1),
            desc=f"Episode: {self.episode}, Reward: ?")
        for episode in pbar:
            self.episode = episode

            self.model = self.model.train()
            episode_reward = self.train_episode()
            episode_rewards.append(episode_reward)

            self.episode_rewards.append(self.env.episode_reward)
            self.episode_driving_rewards.append(
                self.env.episode_driving_reward)
            self.episode_moral_rewards.append(
                self.env.episode_moral_reward)
            self.episode_finished.append(self.env.finished)

            if self.episode % self.episodes_per_log == 0:
                average_episode_reward = np.mean(
                    episode_rewards[-self.metrics_average_len:])

                pbar.set_description(
                    f"Average Reward: {average_episode_reward:.3g}")

                if self.writer is not None:
                    self.writer.add_scalar(
                        'average_episode_reward',
                        average_episode_reward,
                        global_step=self.global_step)
                    self.writer.add_scalar(
                        'epsilon',
                        self.epsilon(),
                        global_step=self.global_step)

            if self.save_path and (
                    self.episode % self.episodes_per_save == 0
                    or self.episode == self.num_episodes):
                self.save_model()

    def train_episode(self):
        episode_reward = 0
        losses = []

        state = self.env.reset()
        state = torch.tensor(state, dtype=torch.float, device=self.device)

        for step in range(1, self.episode_max_len + 1):
            self.model = self.model.eval()
            action = self.select_action(state)
            self.model = self.model.train()
            next_state, reward, done, info = self.env.step(action)

            episode_reward += reward

            action = torch.tensor((action,), device=self.device)
            next_state = torch.tensor(
                next_state, dtype=torch.float, device=self.device)
            reward = torch.tensor((reward,), device=self.device)
            done = torch.tensor((done,), dtype=torch.float, device=self.device)

            batch = self.preprocess_data(
                state, action, next_state, reward, done, info)
            state = next_state

            if batch is None:
                if done:
                    break
                else:
                    continue

            loss = self.loss(**batch)
            loss_normalized = loss / self.gradient_accumulation_steps
            loss_normalized.backward()
            losses.append(loss.item())

            if step % self.gradient_accumulation_steps == 0 \
                    or step >= len(self.data_loader_train):
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            if self.global_step % self.steps_per_log == 0:
                average_loss = np.mean(losses[-self.metrics_average_len:])

                if self.writer is not None:
                    self.writer.add_scalar(
                        'average_loss',
                        average_loss,
                        global_step=self.global_step)

            self.global_step += 1

            if done:
                break

        return episode_reward

    def preprocess_data(self, state, action, next_state, reward, done, info):
        transition = {
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
            }
        self.driving_replay_memory.push(transition)

        if self.double_replay and info['moral_state']:
            self.moral_replay_memory.push(transition)

        # Check if replay memory has enough samples
        if len(self.driving_replay_memory) < self.replay_memory_min:
            return None
        else:
            if (not self.double_replay
                    or len(self.moral_replay_memory) < self.replay_memory_min):
                return self.driving_replay_memory.sample(self.batch_size)
            else:
                batch = [
                    self.driving_replay_memory.sample(self.batch_size),
                    self.moral_replay_memory.sample(self.batch_size),
                ]
                batch = {
                    key: torch.cat((batch[0][key], batch[1][key]), 0)
                    for key in batch[0].keys()}
                return batch

    def loss(self, states, actions, rewards, next_states, dones):
        # Sync target model if needed
        if self.episode % self.target_model_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # Get Q(s, a) of the current state and selected action
        model_q_values = torch.gather(
            self.model(states)['rewards'], 1, actions)

        with torch.no_grad():
            # Get max_a(Q(s', a)) of the next state
            if self.double_dqn:
                next_actions = torch.argmax(
                    self.model(next_states)['rewards'], dim=-1, keepdim=True)
                target_model_q_values = torch.gather(
                    self.target_model(next_states)['rewards'], 1, next_actions)
            else:
                target_model_q_values = torch.max(
                    self.target_model(next_states)['rewards'],
                    dim=-1, keepdim=True)[0]

            # Set Q value of terminal states to 0
            mask = 1 - dones
            target_model_q_values = target_model_q_values * mask

            # Calculate target Q values
            target_q_values = rewards + self.gamma * target_model_q_values

        loss = F.smooth_l1_loss(model_q_values, target_q_values)
        return loss

    def select_action(self, state):
        state = torch.unsqueeze(state, 0)

        sample = random.random()
        if sample > self.epsilon():
            with torch.no_grad():
                action = torch.argmax(self.model(state)['rewards']).item()
        else:
            action = random.randrange(self.num_actions)
        return action

    def epsilon(self):
        return self.epsilon_end \
            + (self.epsilon_start - self.epsilon_end) \
            * math.exp(-1.0 * (self.episode - 1) / self.epsilon_decay)

    def save_model(self):
        if self.save_incrementally:
            save_path = self.save_path \
                / f'{self.save_path.name}_{self.episode}'
        else:
            save_path = self.save_path

        self.model.save(save_path)

        logs = {
            'rewards': np.asarray(self.episode_rewards),
            'driving_rewards': np.asarray(self.episode_driving_rewards),
            'moral_rewards': np.asarray(self.episode_moral_rewards),
            'finished': np.asarray(self.episode_finished),
        }
        np.savez_compressed(save_path / 'logs.npz', **logs)
