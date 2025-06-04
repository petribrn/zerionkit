import numpy as np  # pip install numpy

from src.enums.activation import Activation
from src.enums.loss import Loss
from src.enums.problem_type import ProblemType
from src.neural_network import NeuralNetwork


def main():
    input_layer_size = 9  # arbitrary

    neural_network = NeuralNetwork(
        problem_type=ProblemType.BinaryClassification.value,
        input_layer_size=input_layer_size,
        hidden_layers_sizes=[10, 10],
        output_layer_size=9,
        activation=Activation.Sigmoid.value,
        loss=Loss.CrossEntropy.value,
    )

    inputs_1: list[float] = list(np.random.uniform(low=1, high=101, size=input_layer_size))
    outputs_1 = neural_network.forward(inputs=inputs_1)

    # inputs_2 = neural_network.backpropagation()
