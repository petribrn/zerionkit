import numpy as np  # pip install numpy

from src.neural_network import NeuralNetwork

input_layer_size = 3  # arbitrary
hidden_layer_sizes = [4, 3]  # arbitrary
output_layer_size = 1  # arbitrary


def main():
    neural_network = NeuralNetwork(
        input_layer_size=input_layer_size,
        hidden_layers_sizes=hidden_layer_sizes,
        output_layer_size=output_layer_size,
        problem_type='binary_class',
        activation='sigmoid',
        loss='cross_entropy',
    )

    # TODO: Populate entries with real data from external dataset
    # Adds 1 for the bias
    inputs = list(np.random.uniform(low=1, high=101, size=input_layer_size))
    y_predict_probability = neural_network.forward_pass(inputs=inputs)

    # y_predict = [(0 if  < 0.5 else 1) for y]

    # TODO: Calculate error (y - ŷ)
    # inputs_2 = neural_network.back_propagation(outputs)


if __name__ == '__main__':
    main()
