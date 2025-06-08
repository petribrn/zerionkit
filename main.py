import numpy as np  # pip install numpy

from src.enums.activation import Activation
from src.enums.loss import Loss
from src.enums.problem_type import ProblemType
from src.neural_network import NeuralNetwork

input_layer_size = 3  # arbitrary
hidden_layer_size = 4  # arbitrary
output_layer_size = 2  # arbitrary

def main():
    neural_network = NeuralNetwork(
        problem_type=ProblemType.BinaryClassification.value,
        input_layer_size=input_layer_size,
        hidden_layers_sizes=[hidden_layer_size],
        output_layer_size=output_layer_size,
        activation=Activation.Sigmoid.value,
        loss=Loss.CrossEntropy.value,
    )

    inputs: list[float] = list(np.random.uniform(low=1, high=101, size=input_layer_size))
    for i, input in enumerate(inputs):
        print(f"x{i}: {input}")

    outputs = neural_network.forward(inputs=inputs)

    # inputs_2 = neural_network.backpropagation()


if __name__ == '__main__':
    main()
