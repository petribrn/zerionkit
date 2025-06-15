from src.data_handler import DataHandler
from src.neural_network import NeuralNetwork

# input_layer_size = 3 # arbitrary
hidden_layer_sizes = [4, 3]  # arbitrary
output_layer_size = 1  # arbitrary

binary_class_threshold = 0.5  # arbitrary


def main():
    x: list[list[float]] = [[]]
    y_targets: list[float] = []

    neural_network = NeuralNetwork(
        input_layer_size=len(x[0]),
        hidden_layers_sizes=hidden_layer_sizes,
        output_layer_size=output_layer_size,
        problem_type='binary_class',
        activation='sigmoid',
        loss='cross_entropy',
    )

    acc_error = 0
    for i, y_target in enumerate(y_targets):
        y_predict_probability = neural_network.forward_pass(x=x[i])[0]
        y_predict = 0 if y_predict_probability < binary_class_threshold else 1

        # TODO: Calculate error (y - ŷ)
        obs_error = y_target - y_predict
        acc_error += obs_error


if __name__ == '__main__':
    data_handler = DataHandler(
        problem_type='multi_class',
        data_source_filename='students.csv',
        y_target_column_name='Target',
    )

    print(data_handler.dataframe.head())
    # main()
