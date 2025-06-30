from src.data_handler import DataHandler
from src.zerion_nn import ZerionNN
from src.layer import Layer
import matplotlib.pyplot as plt


def regression_example():
    training_data_handler = DataHandler(
        dataset_type='training',
        problem_type='regression',
        dataset_name='bike',
        y_target_columns='cnt',
    )
    training_inputs, training_y_targets = training_data_handler.processed_data

    validation_data_handler = DataHandler(
        dataset_type='validation',
        problem_type='regression',
        dataset_name='bike',
        y_target_columns='cnt',
    )
    validation_inputs, validation_y_targets = validation_data_handler.processed_data

    test_data_handler = DataHandler(
        dataset_type='test',
        problem_type='regression',
        dataset_name='bike',
        y_target_columns='cnt',
    )
    test_inputs, test_y_targets = test_data_handler.processed_data

    # Neural Network Instance with problem type, layers, loss, learning rate and epochs definition
    neural_network = ZerionNN(
        problem_type='regression',
        layers=[
            Layer(size=len(training_inputs[0]), activation='linear'),
            Layer(size=8, activation='relu'),
            Layer(size=len(training_y_targets.keys()), activation='linear'),
        ],
        loss='square_error',
        learning_rate=0.001,
        epochs=10,
    )

    # Train model
    metrics = neural_network.train(
        inputs=training_inputs,
        y_targets=training_y_targets,
    )

    errors = metrics['errors']
    epochs_range = range(len(errors))

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot Error
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Error (Loss)', color=color)
    ax1.plot(epochs_range, errors, color=color, label='Training Error (Loss)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)
    ax1.legend(loc='upper right')
    fig.suptitle('Training Error vs. Epochs')
    fig.tight_layout()

    # Evaluate with validation dataset
    neural_network.evaluate(
        inputs=validation_inputs,
        y_targets=validation_y_targets,
        y_scaler=validation_data_handler.y_scaler,
    )

    # Evaluate with test dataset
    neural_network.evaluate(
        inputs=test_inputs,
        y_targets=test_y_targets,
        y_scaler=test_data_handler.y_scaler,
    )

    plt.show()
