import numpy as np

from src.neural_network import NeuralNetwork

input_layer_size = 3  # arbitrary
hidden_layer_sizes = [4, 3]  # arbitrary
output_layer_size = 1  # arbitrary


def main():
    # data_handler = DataHandler(
    #     problem_type='multi_class',
    #     data_source_filename='students.csv',
    #     y_target_column_name='Target',
    # )

    x: list[list[float]] = [
        list(np.random.uniform(low=1, high=101, size=input_layer_size)) for _ in range(0, 10)
    ] # inputs
    y_targets: list[float] = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0] # target outputs

    print(x)
    print(y_targets)

    neural_network = NeuralNetwork(
        input_layer_size=input_layer_size,
        hidden_layers_sizes=hidden_layer_sizes,
        output_layer_size=output_layer_size,
        problem_type='binary_class',
        activation='sigmoid',
        loss='binary_cross_entropy',
    )

    for n, y_target in enumerate(y_targets):
        neural_network.train_on_iteration(
            x=x[n],
            y_target=y_targets[n],
        )


if __name__ == '__main__':
    main()
