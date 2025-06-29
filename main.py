from src.data_handler import DataHandler
from src.zerion_nn import ZerionNN
from src.layer import Layer


def main():
    training_data_handler = DataHandler(
        dataset_type='training',
        problem_type='binary_class',
        dataset_name='mushroom',
        y_target_columns='poisonous',
    )

    # training_data_handler = DataHandler(
    #     dataset_type = 'training',
    #     problem_type='regression',
    #     dataset_name='bike',
    #     y_target_columns='cnt',
    # )

    # training_data_handler = DataHandler(
    #     dataset_type = 'training',
    #     problem_type='multi_class',
    #     dataset_name = 'students',
    #     y_target_columns=['target_Dropout', 'target_Enrolled', 'target_Graduate'],
    # )

    validation_data_handler = DataHandler(
        dataset_type='validation',
        problem_type='binary_class',
        dataset_name='mushroom',
        y_target_columns='poisonous',
    )

    test_data_handler = DataHandler(
        dataset_type='test',
        problem_type='binary_class',
        dataset_name='mushroom',
        y_target_columns='poisonous',
    )

    training_inputs, training_y_targets = training_data_handler.processed_data

    neural_network = ZerionNN(
        problem_type='binary_class',
        layers=[
            Layer(size=len(training_inputs[0]), activation='sigmoid'),
            Layer(size=4, activation='relu'),
            Layer(size=8, activation='relu'),
            Layer(size=1, activation='sigmoid'),
        ],
        loss='binary_cross_entropy',
        learning_rate=0.35,
        epochs=4,
    )

    errors = neural_network.train(
        inputs=training_inputs,
        y_targets=training_y_targets,
    )

    neural_network.evaluate(
        inputs=validation_data_handler.processed_data[0],
        y_targets=validation_data_handler.processed_data[1],
    )

    # Testing XOR
    # output = neural_network.predict([1,1])
    # output = 1 if output[0] > 0.5 else 0
    # print(f'Output: {output}')


if __name__ == '__main__':
    main()
