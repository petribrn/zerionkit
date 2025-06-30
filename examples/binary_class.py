from src.data_handler import DataHandler
from src.zerion_nn import ZerionNN
from src.layer import Layer
import matplotlib.pyplot as plt


def binary_class_example():
    training_data_handler = DataHandler(
        dataset_type='training',
        problem_type='binary_class',
        dataset_name='mushroom',
        y_target_columns='poisonous',
    )
    training_inputs, training_y_targets = training_data_handler.processed_data

    validation_data_handler = DataHandler(
        dataset_type='validation',
        problem_type='binary_class',
        dataset_name='mushroom',
        y_target_columns='poisonous',
    )
    validation_inputs, validation_y_targets = validation_data_handler.processed_data

    test_data_handler = DataHandler(
        dataset_type='test',
        problem_type='binary_class',
        dataset_name='mushroom',
        y_target_columns='poisonous',
    )
    test_inputs, test_y_targets = test_data_handler.processed_data

    # Neural Network Instance with problem type, layers, loss, learning rate and epochs definition
    neural_network = ZerionNN(
        problem_type='binary_class',
        layers=[
            Layer(size=len(training_inputs[0]), activation='sigmoid'),
            Layer(size=4, activation='relu'),
            Layer(size=8, activation='relu'),
            Layer(size=len(training_y_targets.keys()), activation='sigmoid'),
        ],
        loss='binary_cross_entropy',
        learning_rate=0.01,
        epochs=100,
    )

    # Train model
    metrics = neural_network.train(
        inputs=training_inputs,
        y_targets=training_y_targets,
    )

    errors = metrics['errors']
    accuracies = metrics['accuracies']
    epochs_range = range(len(errors))

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot Training Error
    ax1.plot(epochs_range, errors, 'r-', label='Training Error (Loss)')
    ax1.set_ylabel('Error (Loss)')
    ax1.set_title('Training Error and Accuracy Over Epochs')
    ax1.grid(True)
    ax1.legend()

    # Plot Training Accuracy
    ax2.plot(epochs_range, accuracies, 'b-', label='Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    ax2.legend()

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
