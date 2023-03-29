import numpy as np
from datetime import datetime
import torch


def smooth(to_smooth):
    n_episodes = len(to_smooth)
    smoothing_window = n_episodes // 10
    smoothed = np.convolve(to_smooth, np.ones(smoothing_window)/smoothing_window, mode='valid')
    episodes = np.arange(n_episodes)[smoothing_window//2:-smoothing_window//2+1]
    return episodes, smoothed


class Timer():
    """A class for timing code execution."""
    def __init__(self):
        self.start = datetime.now()
        self.end = None
        self.elapsed_time = None

    def stop_timer(self):
        self.end = datetime.now()
        self.elapsed_time = self.end - self.start
        print('Execution time: {}'.format(self.elapsed_time))

def make_torch_float32(observation):
    if torch.is_tensor(observation):
        if observation.dtype != torch.float32:
            state = observation.type(torch.float32)
        else:
            state = observation
    else:
        state = torch.tensor(observation, dtype=torch.float32)
    return state