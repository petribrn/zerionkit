from src.data_handler import DataHandler
from src.zerion_nn import ZerionNN
from src.layer import Layer


def main(processed_data: tuple[list[list[float]], dict[str, list[float]]]):
    inputs, y_targets = processed_data

    neural_network = ZerionNN(
        problem_type='binary_class',
        layers=[
            Layer(size=len(inputs[0]), activation='sigmoid'),
            Layer(size=2, activation='relu'),
            Layer(size=1, activation='sigmoid'),
        ],
        loss='binary_cross_entropy',
        learning_rate=0.5,
        epochs=400,
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
    data_handler = DataHandler(
        problem_type='binary_class',
        data_source_filename='mushroom_converted.csv',
        y_target_columns='poisonous',
        auto_process=True,
    )

    # REGRESSION
    # data_handler = DataHandler(
    #     problem_type='regression',
    #     data_source_filename='bike_converted.csv',
    #     y_target_columns='cnt',
    #     auto_process=True,
    # )

    # MULTI_CLASS
    # data_handler = DataHandler(
    #     problem_type='multi_class',
    #     data_source_filename='students_converted.csv',
    #     y_target_columns=['target_Dropout', 'target_Enrolled', 'target_Graduate'],
    #     auto_process=True,
    # )

    main(processed_data=data_handler.validation_data)