from src.data_handler import DataHandler
from src.zerion_nn import ZerionNN
from src.layer import Layer


def main(processed_data: tuple[list[list[float]], dict[str, list[float]]]):
    inputs, y_targets = processed_data

    neural_network = ZerionNN(
        problem_type='regression',
        layers=[
            Layer(size=len(inputs[0]), activation='linear'),
            Layer(size=2, activation='relu'),
            Layer(size=1, activation='linear'),
        ],
        loss='square_error',
        learning_rate=0.35,
        epochs=4,
    )

    neural_network.train(
        inputs=inputs,
        y_target=y_targets,
    )

    # Testing XOR
    # output = neural_network.test([1,1])
    # output = 1 if output[0] > 0.5 else 0
    # print(f'Output: {output}')


if __name__ == '__main__':
    # BINARY_CLASS
    # data_handler = DataHandler(
    #     problem_type='binary_class',
    #     dataset_name = 'mushroom',
    #     dataset_type = 'training',
    #     y_target_columns='poisonous',
    # )

    # REGRESSION
    data_handler = DataHandler(
        problem_type='regression',
        dataset_name='bike',
        dataset_type='training',
        y_target_columns='cnt',
    )

    # MULTI_CLASS
    # data_handler = DataHandler(
    #     problem_type='multi_class',
    #     dataset_name = 'students',
    #     dataset_type = 'training',
    #     y_target_columns=['target_Dropout', 'target_Enrolled', 'target_Graduate'],
    # )

    # main(processed_data=data_handler.processed_data)
