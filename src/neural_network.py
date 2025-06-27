import math
from typing import Literal, Callable

import numpy as np


class NeuralNetwork:
    def __init__(
            self,
            input_layer_size: int,
            hidden_layers_sizes: list[int],
            output_layer_size: int,
            problem_type: Literal['regression', 'binary_class', 'multi_class'],
            activation: Literal['linear', 'sigmoid', 'softmax'],
            loss: Literal['square_error', 'binary_cross_entropy', 'cross_entropy'],
            learning_rate: float,
    ):
        self.input_layer_size = input_layer_size
        self.hidden_layers_sizes = hidden_layers_sizes
        self.output_layer_size = output_layer_size
        self.all_layers_sizes = [self.input_layer_size] + self.hidden_layers_sizes + [self.output_layer_size]

        self.problem_type = problem_type
        self.activation_function = self.Activation.get_activation_func(activation)
        self.derivative_activation_function = self.Activation.get_derivative_activation_func(activation)
        self.loss = loss

        if learning_rate <= 0.0 or learning_rate > 1.0:
            raise Exception()
        self.learning_rate = learning_rate

        self.x: list[list[float]] = []
        self.weights: list[list[list[float]]] = []
        self.d_weights: list[list[list[float]]] = []

    def train(
            self,
            x: list[list[float]],
            y_target: dict[str, list[float]],
    ):
        if len(x) == 0 or len(y_target.values()) == 0:
            raise Exception()

        n = len(x)

        for i in range(0, n):
            y_predict: list[float] = self.__forward_pass(x=x[i])
            y_target_for_i = [dict(zip(y_target.keys(), values)) for values in zip(*y_target.values())][i]

            # output layer error gradients
            d_y: list[float] = self.__apply_loss(
                y_predict=y_predict,
                y_target=[y_target_for_i[sorted(y_target_for_i.keys())[0]]],
            )

            self.__back_propagation(d_y=d_y)
            self.__update_weights()

    # - Apply a dot product between layers arrays and their respective weight matrix (wT).
    # - Apply activation function to each of the dot product results, to make it non-linear (if activation != Linear);
    # - Repeat this process until it gets the output layer outputs;
    def __forward_pass(self, x: list[float]) -> list[float]:
        y_predict: list[float] = x

        i = 0
        while i < len(self.all_layers_sizes) - 1:
            y_predict = [1] + y_predict
            self.x.append(y_predict)

            # Adds 1 for the bias
            current_layer_size = self.all_layers_sizes[i] + 1
            next_layer_size = self.all_layers_sizes[i + 1]

            next_layer_weights = self.__get_next_layer_weights(
                current_layer_size=current_layer_size,
                next_layer_size=next_layer_size,
            )

            self.weights.append(next_layer_weights)

            v: list[float] = np.dot(next_layer_weights, y_predict).tolist()
            normalized_v: list[float] = [vi / 1000 for vi in v]
            self.x.append(normalized_v)

            y_predict = self.activation_function(normalized_v)

            i += 1

        return y_predict

    def __get_next_layer_weights(
            self,
            current_layer_size: int,
            next_layer_size: int,
    ) -> list[list[float]]:

        # size = matrix (rows, columns)
        next_layer_weights: list[list[float]] = (
            np.random.uniform(
                low=0.1,
                high=1,
                size=(next_layer_size, current_layer_size)
            ).tolist()
        )

        return next_layer_weights

    # - Calculate the output layer error gradient given y predict and y target, to get the y value
    # in which the error most increases.
    def __apply_loss(
            self,
            y_predict: list[float],
            y_target: list[float],
    ) -> list[float]:
        d_error = []

        match self.problem_type:
            case 'regression':
                d_error = self.Loss.Mse.derivative(
                    y_predict=y_predict[0],
                    y_target=y_target[0],
                )
            case 'binary_class':
                d_error = self.Loss.BinaryCrossEntropy.derivative(
                    y_predict=y_predict[0],
                    y_target=y_target[0],
                )
            case 'multi_class':
                d_error = self.Loss.MultiClassCrossEntropy.derivative(
                    y_predict=y_predict,
                    y_target=y_target,
                )

        return d_error

    # - Backpropagate the y error gradients to previous layers
    # - Calculate the new bias and weights for next iteration

    # self.weights[-1].append([x * 2 for x in self.weights[-1][0]])
    # print(self.weights[-1])
    #
    # for i in self.weights[-1]:
    #     print(i)
    def __back_propagation(self, d_y: list[float]):
        d_y_linear: list[float] = d_y

        for i in reversed(range(0, len(self.all_layers_sizes))):
            x_derivatives: list[float] = self.derivative_activation_function(self.x[i])

            # Activation layer
            d_y_activ: list[float] = [d_y_i * last_x_d for d_y_i, last_x_d in zip(d_y_linear, x_derivatives)]

            # Transpose weights for back-propagation
            w_t: list[list[float]] = np.transpose(self.weights[i]).tolist()

            # Save weight error gradients for later update
            d_w: list[list[float]] = np.outer(self.x[i - 1], d_y_activ).tolist()
            self.d_weights.append(d_w)

            # Calculate the previous layer error gradient to pass to the previous layer
            d_y_linear: list[float] = np.dot(w_t, d_y_activ).tolist()

    def __update_weights(self):
        for c1, k in enumerate(self.weights):
            for c2, i in enumerate(k):
                for c3, j in enumerate(i):
                    self.weights[c1][c2][c3] -= self.learning_rate * self.d_weights[c1][c2][c3]

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
            return [1 / (1 + math.exp(-xi)) for xi in x]

        @classmethod
        def sigmoid_derivative(cls, x: list[float]) -> list[float]:
            sigmoids = cls.sigmoid(x)
            return [sigmoid * (1 - sigmoid) for sigmoid in sigmoids]

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
        def get_activation_func(cls, selected_activation_func: Literal['linear', 'sigmoid', 'softmax']) -> Callable[
            [list[float]], list[float]]:
            match selected_activation_func:
                case 'linear':
                    return cls.linear
                case 'sigmoid':
                    return cls.sigmoid
                case 'softmax':
                    return cls.softmax
                case _:
                    return None

        @classmethod
        def get_derivative_activation_func(cls, selected_activation_func: Literal['linear', 'sigmoid', 'softmax']) -> \
                Callable[[list[float]], list[float]]:
            match selected_activation_func:
                case 'linear':
                    return cls.linear_derivative
                case 'sigmoid':
                    return cls.sigmoid_derivative
                case 'softmax':
                    return cls.softmax_derivative
                case _:
                    return None

    class Loss:
        class Mse:
            @staticmethod
            def main(y_predict: float, y_target: float) -> list[float]:
                return [(y_target - y_predict) ** 2]

            @staticmethod
            def derivative(y_predict: float, y_target: float) -> list[float]:
                return [y_predict - y_target]

        class BinaryCrossEntropy:
            @staticmethod
            def main(y_predict: float, y_target: float) -> list[float]:
                y_predict_norm = normalize_y_for_classification(y_predict)

                return [
                    - (y_target * math.log(y_predict_norm) + (1 - y_target) * math.log(1 - y_predict_norm))
                ]

            @staticmethod
            def derivative(y_predict: float, y_target: float) -> list[float]:
                y_predict_norm = normalize_y_for_classification(y_predict)

                return [
                    - (y_target / y_predict_norm) + (1 - y_target) / (1 - y_predict_norm)
                ]

        class MultiClassCrossEntropy:
            @staticmethod
            def main(y_predict: list[float], y_target: list[float]) -> list[float]:
                y_predict_norm = [normalize_y_for_classification(y_p_r) for y_p_r in y_predict]

                return [
                    - sum(
                        y_t * math.log(y_p) for y_t, y_p in zip(y_target, y_predict_norm)
                    )
                ]

            @staticmethod
            def derivative(y_predict: list[float], y_target: list[float]) -> list[float]:
                y_predict_norm = [normalize_y_for_classification(y_p_r) for y_p_r in y_predict]

                return [y_p - y_t for y_p, y_t in zip(y_predict_norm, y_target)]


def normalize_y_for_classification(y_predict: float) -> float:
    eps = 1e-15  # avoid log(0)
    return max(min(y_predict, 1 - eps), eps)
