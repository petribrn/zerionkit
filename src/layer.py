from typing import Literal, Callable
import math
import numpy as np

ACTIVATION_FUNCTION_TYPES = Literal['linear', 'sigmoid', 'softmax', 'relu']


class Layer:
    def __init__(
            self,
            size: int,
            activation: ACTIVATION_FUNCTION_TYPES,
    ):
        self.size = size
        self.activation_function = self.Activation.get_activation_func(activation)
        self.derivative_activation_function = self.Activation.get_derivative_activation_func(activation)

    def __iter__(self):
        # Used to get class as dict
        for key, value in self.__dict__.items():
            yield key, value

    class Activation:
        """
            Activation Class

            Implements activation functions methods of the neural network core.
        """

        @classmethod
        def linear(cls, x: list[float]) -> list[float]:
            return x

        @classmethod
        def linear_derivative(cls, x: list[float]) -> list[float]:
            return [1.0] * len(x)

        @classmethod
        def sigmoid(cls, x: list[float]) -> list[float]:
            return [float(1.0 / (1.0 + np.exp(-xi))) for xi in x]

        @classmethod
        def sigmoid_derivative(cls, x: list[float]) -> list[float]:
            sigmoids = cls.sigmoid(x)
            return [sigmoid * (1.0 - sigmoid) for sigmoid in sigmoids]

        @classmethod
        def softmax(cls, x: list[float]) -> list[float]:
            exp_x = [math.exp(xi - max(x)) for xi in x]
            sum_exp = sum(exp_x)

            return [xi / sum_exp for xi in exp_x]

        @classmethod
        def softmax_derivative(cls, x: list[float]) -> list[list[float]]:
            softmax = cls.softmax(x)
            n = len(softmax)

            jacobian_matrix = []

            for i in range(n):
                row = []

                for j in range(n):
                    if i == j:
                        row.append(softmax[i] * (1 - softmax[i]))
                    else:
                        row.append(-softmax[i] * softmax[j])
                jacobian_matrix.append(row)

            return jacobian_matrix

        @classmethod
        def relu(cls, x: list[float]) -> list[float]:
            return [max(0.0, xi) for xi in x]

        @classmethod
        def relu_derivative(cls, x: list[float]) -> list[float]:
            return [1.0 if xi > 0 else 0.0 for xi in x]

        @classmethod
        def get_activation_func(cls, selected_activation_func: ACTIVATION_FUNCTION_TYPES) -> Callable[
            [list[float]], list[float]]:
            match selected_activation_func:
                case 'linear':
                    return cls.linear
                case 'sigmoid':
                    return cls.sigmoid
                case 'softmax':
                    return cls.softmax
                case 'relu':
                    return cls.relu
                case _:
                    return None

        @classmethod
        def get_derivative_activation_func(cls, selected_activation_func: ACTIVATION_FUNCTION_TYPES) -> \
                Callable[[list[float]], (list[float] | list[list[float]])]:
            match selected_activation_func:
                case 'linear':
                    return cls.linear_derivative
                case 'sigmoid':
                    return cls.sigmoid_derivative
                case 'softmax':
                    return cls.softmax_derivative
                case 'relu':
                    return cls.relu_derivative
                case _:
                    return None
