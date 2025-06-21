import math
from typing import Literal

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
        self.problem_type = problem_type
        self.activation_function = self.Activation.get_activation_func(activation)
        self.loss = loss

        if learning_rate < 0.0 or learning_rate < 1.0:
            raise Exception()
        self.learning_rate = learning_rate

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
            y_error_gradients: list[float] = self.__apply_loss(
                y_predict=y_predict,
                y_target=[y_target_for_i[sorted(y_target_for_i.keys())[0]]],
            )

            self.__back_propagation(y_error_gradients=y_error_gradients)

    # - Apply a dot product between layers arrays and their respective weight matrix (wT).
    # - Apply activation function to each of the dot product results, to make it non-linear (if activation != Linear);
    # - Repeat this process until it gets the output layer outputs;
    def __forward_pass(self, x: list[float]) -> list[float]:
        print("Input layer neurons")
        for c, input in enumerate(x):
            print(f"x{c}: {input}")

        print('-----')

        y_predict: list[float] = x
        all_layers_sizes = [self.input_layer_size] + self.hidden_layers_sizes + [self.output_layer_size]

        i = 0
        while i < len(all_layers_sizes) - 1:
            y_predict = [1] + y_predict

            # Adds 1 for the bias
            current_layer_size = all_layers_sizes[i] + 1
            next_layer_size = all_layers_sizes[i + 1]

            next_layer_weights = self.__get_next_layer_weights(
                current_layer_size=current_layer_size,
                next_layer_size=next_layer_size,
            )

            v: list[float] = np.dot(next_layer_weights, y_predict)
            normalized_v: list[float] = [vi / 1000 for vi in v]

            # for j, v_number in enumerate(normalized_v):
            #     print(f"v{j}: {v_number}")
            #
            # print("-----")

            y_predict = self.activation_function(normalized_v)
            print(f'ŷ: {y_predict}')
            print("-----")

            i += 1

        return y_predict

    def __get_next_layer_weights(
            self,
            current_layer_size: int,
            next_layer_size: int,
    ) -> list[list[float]]:
        print("Generating layer weights")

        # size = matrix (rows, columns)
        next_layer_weights: list[list[float]] = (
            np.random.uniform(
                low=0.1,
                high=1,
                size=(next_layer_size, current_layer_size)
            ).tolist()
        )

        # for i, hidden in enumerate(next_layer_weights):
        #     for j, input_weight in enumerate(hidden):
        #         if j == 0:
        #             print(f"b{i}: {input_weight}")
        #         else:
        #             print(f"w{i}{j}: {input_weight}")
        #
        #     print("-----")

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
    def __back_propagation(self, y_error_gradients: list[float]) -> list[float]:
        ...

    class Activation:
        """
            Activation Class

            Implements activation functions methods of the neural network core.
        """

        @staticmethod
        def linear(x: list[float]) -> list[float]:
            return x

        @staticmethod
        def sigmoid(x: list[float]) -> list[float]:
            return [1 / (1 + math.exp(-xi)) for xi in x]

        @staticmethod
        def softmax(x: list[float]) -> list[float]:
            exp_x = [math.exp(xi - max(x)) for xi in x]
            return [xi / sum(exp_x) for xi in exp_x]

        @classmethod
        def get_activation_func(cls, selected_activation_func: Literal['linear', 'sigmoid', 'softmax']):
            match selected_activation_func:
                case 'linear':
                    return cls.linear
                case 'sigmoid':
                    return cls.sigmoid
                case 'softmax':
                    return cls.softmax
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

                return [
                    - (y_t / y_p) for y_t, y_p in zip(y_target, y_predict_norm)
                ]


def normalize_y_for_classification(y_predict: float) -> float:
    eps = 1e-15  # avoid log(0)
    return max(min(y_predict, 1 - eps), eps)
