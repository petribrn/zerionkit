import numpy as np
import math
from typing import Literal


class NeuralNetwork:
    def __init__(
            self,
            input_layer_size: int,
            hidden_layers_sizes: list[int],
            output_layer_size: int,
            problem_type: Literal['regression', 'binary_class', 'multi_class'],
            activation: Literal['linear', 'sigmoid', 'softmax'],
            loss: Literal['mse', 'cross_entropy'],
    ):
        self.input_layer_size = input_layer_size
        self.hidden_layers_sizes = hidden_layers_sizes
        self.output_layer_size = output_layer_size
        self.problem_type = problem_type
        self.activation_function = self.Activation.get_activation_func(activation)
        self.loss = loss

    # - Apply dot product (produto escalar) between input array (column size 1, row size "input_layer_size") and weight
    # matrix (wT) (row size "input_layer_size", column size "hidden_layers_sizes[i]" or "output_layer_size"). Each weight is a connection
    # between an input neuron and a hidden (or output, if zero hidden layers) neuron. The input array starts with a 1
    # for the Bias b. The dot product will output an array of linear float values (column size 1, row size "hidden_layers_sizes[i]" or "output_layer_size").

    # - Apply activation function to each of the dot product results, to make it non-linear (if activation != Linear);
    # - Repeat this process until it gets the output layer outputs;
    def forward_pass(self, inputs: list[float]) -> list[float]:
        print("Input layer neurons")
        for i, input in enumerate(inputs):
            print(f"x{i}: {input}")

        print('-----')

        # size = matrix (rows, columns)
        input_to_hidden1_initial_weights: list[list[float]] = (
            np.random.uniform(
                low=1,
                high=101,
                size=(self.hidden_layers_sizes[0], self.input_layer_size + 1) # Adds 1 for the bias
            ).tolist()
        )

        for i, hidden in enumerate(input_to_hidden1_initial_weights):
            for j, input_weight in enumerate(hidden):
                if j == 0:
                    print(f"b{i}: {input_weight}")
                else:
                    print(f"w{i}{j}: {input_weight}")

            print("-----")

        v: list[float] = np.dot(input_to_hidden1_initial_weights, inputs)
        normalized_v = [vi / 1000 for vi in v]

        for i, v_number in enumerate(normalized_v):
            print(f"v{i}: {v_number}")

        print("-----")

        y = self.activation_function(normalized_v)
        print(y)

        return y

    # - Calculate loss based on the predicted y and correct y
    # - Calculate gradient descent to get the direction (value) where the error decreases
    # - Calculate the new bias and weights for next iteration
    # - Backpropagate the weights
    def back_propagation(self, y_predict, y_target):
        ...

    # Run all together
    def train(self, X, Y, epochs, lr):
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
