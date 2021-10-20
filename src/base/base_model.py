from abc import ABC, abstractmethod
import json
from pathlib import Path

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def config(self):
        pass

    @abstractmethod
    def init_parameters(self):
        pass

    @abstractmethod
    def reset_parameters(self):
        pass

    def parameter_dicts(self):
        parameter_dicts = [{'params': self.parameters()}]
        return parameter_dicts

    def save(self, model_path, **config_override):
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)
        config_path = model_path / 'config.json'
        state_dict_path = model_path / 'state_dict.pt'

        config = self.config()
        config.update(config_override)
        with config_path.open('w') as file:
            json.dump(config, file, indent=4)

        torch.save(self.state_dict(), state_dict_path)

    @classmethod
    def load(cls, model_path, **config_override):
        model_path = Path(model_path)
        config_path = model_path / 'config.json'
        state_dict_path = model_path / 'state_dict.pt'

        with config_path.open() as file:
            config = json.load(file)
        config.update(config_override)

        model = cls(**config)
        model.load_state_dict(torch.load(state_dict_path))
        return model
