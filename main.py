import numpy as np # pip install numpy
from src.neural_network import NeuralNetwork, ProblemType, Activation, Loss


def main():
    input_layer_size = 9 # arbitrary

    neural_network = NeuralNetwork(
        problem_type=ProblemType.BinaryClassification.value,
        input_layer_size=input_layer_size,
        hidden_layers_sizes=[10, 10],
        output_layer_size=9,
        activation=Activation.Sigmoid.value,
        loss=Loss.BinaryCrossEntropy.value,
    )

    inputs_1: list[float] = list(np.random.uniform(low=1, high=101, size=input_layer_size))
    outputs_1 = neural_network.forward(inputs=inputs_1)

    # inputs_2 = neural_network.backpropagation()
