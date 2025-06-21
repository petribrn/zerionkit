import numpy as np

from src.neural_network import NeuralNetwork
from src.data_handler import DataHandler

input_layer_size = 3  # arbitrary
hidden_layer_sizes = [4, 3]  # arbitrary
output_layer_size = 1  # arbitrary


def main(processed_data: tuple[list[list[float]], dict[str, list[float]]]):
    x, y_targets = processed_data

    # x: list[list[float]] = [
    #     list(np.random.uniform(low=1, high=101, size=input_layer_size)) for _ in range(0, 10)
    # ] # inputs
    # y_targets: list[float] = [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0] # target outputs

    print(y_targets)

    # neural_network = NeuralNetwork(
    #     input_layer_size=input_layer_size,
    #     hidden_layers_sizes=hidden_layer_sizes,
    #     output_layer_size=output_layer_size,
    #     problem_type='binary_class',
    #     activation='sigmoid',
    #     loss='binary_cross_entropy',
    #     learning_rate=0.5,
    # )
    #
    # for n, y_target in enumerate(y_targets):
    #     neural_network.train_on_iteration(
    #         x=x[n],
    #         y_target=y_targets[n],
    #     )


if __name__ == '__main__':

    # BINARY_CLASS
    # data_handler = DataHandler(
    #     problem_type='binary_class',
    #     data_source_filename='mushroom_converted.csv',
    #     y_target_columns='poisonous',
    # )

    # REGRESSION
    data_handler = DataHandler(
        problem_type='regression',
        data_source_filename='bike_converted.csv',
        y_target_columns='cnt',
    )

    # MULTI_CLASS
    # data_handler = DataHandler(
    #     problem_type='multi_class',
    #     data_source_filename='students_converted.csv',
    #     y_target_columns=['target_Dropout', 'target_Enrolled', 'target_Graduate'],
    # )

    main(processed_data=data_handler.processed_data)
