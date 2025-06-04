from enum import Enum


class Activation(Enum):
    Linear = 0  # (no activation)
    Sigmoid = 1
    Softmax = 2