
from enum import Enum


class Loss(Enum):
    Mse = 0  # mean square error
    BinaryCrossEntropy = 1
    CrossEntropy = 2