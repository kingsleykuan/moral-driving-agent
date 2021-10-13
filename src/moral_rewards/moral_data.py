import numpy as np
from torch.utils.data import Dataset


class MoralMachineDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.data_not_saved = data['data_not_saved']
        self.data_saved = data['data_saved']

    def __len__(self):
        return len(self.data_saved)

    def __getitem__(self, idx):
        data_not_saved = self.data_not_saved[idx].astype(np.float32)
        data_saved = self.data_saved[idx].astype(np.float32)

        data = {
            'data_not_saved': data_not_saved,
            'data_saved': data_saved,
        }

        return data
