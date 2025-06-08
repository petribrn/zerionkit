import numpy as np  # pip install numpy
from src.neural_network import NeuralNetwork

input_layer_size = 3  # arbitrary
hidden_layer_size = 4  # arbitrary
output_layer_size = 2  # arbitrary

def main():
    neural_network = NeuralNetwork(
        input_layer_size=input_layer_size,
        hidden_layers_sizes=[hidden_layer_size],
        output_layer_size=output_layer_size,
        problem_type='binary_class',
        activation='sigmoid',
        loss='cross_entropy',
    )

    # TODO: Populate entries with real data from external dataset
    # Adds 1 for the bias
    inputs = [1] + list(np.random.uniform(low=1, high=101, size=input_layer_size))
    outputs = neural_network.forward_pass(inputs=inputs)

    # inputs_2 = neural_network.back_propagation(outputs)

if __name__ == '__main__':
    main()
