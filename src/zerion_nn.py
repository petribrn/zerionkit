from typing import Literal

from sklearn.preprocessing import StandardScaler

from src.layer import Layer
from src.loss import Loss

import numpy as np


class ZerionNN:
    def __init__(
            self,
            problem_type: Literal['regression', 'binary_class', 'multi_class'],
            layers: list[Layer],
            loss: Literal['square_error', 'binary_cross_entropy', 'cross_entropy'],
            learning_rate: float,
            epochs: int,
    ):
        self.layers = layers
        self.problem_type = problem_type
        self.loss = loss

        if learning_rate <= 0.0 or learning_rate > 1.0:
            raise Exception("Learning rate must be between 0 and 1.")
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.x: list[list[float]] = []  # before applying activation function
        self.z: list[list[float]] = []  # after applying activation function

        self.weights: list[list[list[float]]] = self.__generate_weights()
        self.d_weights: list[list[list[float]]] = []

    def __generate_weights(self) -> list[list[list[float]]]:
        weights: list[list[list[float]]] = []
        layer_sizes = [l.size for l in self.layers]

        def zip_with_next(lst: list):
            return list(zip(lst, lst[1:]))

        for current_layer_size_with_next in zip_with_next(layer_sizes):
            current_size, next_size = current_layer_size_with_next

            # size = matrix (rows, columns)
            next_layer_weights: list[list[float]] = (
                np.random.uniform(
                    low=-1.0,
                    high=1.0,
                    size=(next_size, current_size + 1)  # Adds 1 for the bias
                ).tolist()
            )

            weights.append(next_layer_weights)

        return weights

    def train(
            self,
            inputs: list[list[float]],
            y_targets: dict[str, list[float]],
    ):
        n = len(inputs)

        if n == 0 or len(y_targets.values()) == 0:
            raise Exception()

        epochs_errors: list[float] = []
        epochs_accuracies: list[float] = []

        for e in range(self.epochs):
            iterations_errors: list[list[float]] = []
            correct_predictions_in_epoch = 0

            for i in range(n):
                self.x = []
                self.d_weights = []

                y_predict: list[float] = self.__forward_pass(inputs=inputs[i])
                y_target_for_i = [dict(zip(y_targets.keys(), values)) for values in zip(*y_targets.values())][i]

                # Only for metrics, does not impact the training process
                error: list[float] = self.__calculate_error(
                    y_predict=y_predict,
                    y_target=list(y_target_for_i.values()),
                )
                iterations_errors.append(error)

                if self.problem_type == 'binary_class':
                    # For binary, apply a 0.5 threshold to the single output neuron
                    pred_class = 1 if y_predict[0] > 0.5 else 0
                    true_class = int(list(y_target_for_i.values())[0])
                    if pred_class == true_class:
                        correct_predictions_in_epoch += 1
                elif self.problem_type == 'multi_class':
                    # For multi-class, find the index of the highest prob
                    pred_class = np.argmax(y_predict)
                    true_class = np.argmax(y_target_for_i.values())
                    if pred_class == true_class:
                        correct_predictions_in_epoch += 1

                # output layer error gradients
                d_y: list[float] = self.__calculate_error_gradient(
                    y_predict=y_predict,
                    y_target=list(y_target_for_i.values()),
                )

                self.__back_propagation(d_y=d_y)
                self.__update_weights()

            mean_error = float(np.mean([err[0] for err in iterations_errors]))
            epoch_accuracy = (correct_predictions_in_epoch / n) * 100 if n > 0 else 0
            accuracy_to_print = f'| Accuracy: {epoch_accuracy}%' if self.problem_type != 'regression' else ''
            print(f'[Epoch {e}] Error: {mean_error}{accuracy_to_print}')

            epochs_errors.append(mean_error)
            epochs_accuracies.append(epoch_accuracy)

        return {
            'errors': epochs_errors,
            'accuracies': epochs_accuracies
        }

    def predict(self, inputs: list[float]) -> list[float]:
        return self.__forward_pass(inputs)

    def evaluate(
            self,
            inputs: list[list[float]],
            y_targets: dict[str, list[float]],
            y_scaler: StandardScaler = None,
    ) -> dict[str, float | int]:
        n = len(inputs)
        all_y_predict = []
        all_y_target = []

        # Get the name of the target variable from the dictionary
        y_target_keys = sorted(y_targets.keys())

        for i in range(n):
            # Predict values using current network weights
            prediction = self.predict(inputs=inputs[i])
            y_target_i = [y_targets[key][i] for key in y_target_keys]

            all_y_predict.append(prediction)
            all_y_target.append(y_target_i)

        metrics = {}
        print(f"\n--- Evaluation Results ---")

        if self.problem_type == 'regression':
            # Flatten the lists for calculation
            y_predict_scaled = np.array(all_y_predict).flatten().reshape(-1,1).tolist()
            y_target_scaled = np.array(all_y_target).flatten().reshape(-1,1).tolist()

            if y_scaler is None:
                raise ValueError("y_scaler must be provided for regression evaluation.")

            y_predict_unscaled = y_scaler.inverse_transform(y_predict_scaled)
            y_target_unscaled = y_scaler.inverse_transform(y_target_scaled)

            # Mean Squared Error (MSE)
            mse = np.mean(Loss.SquareError.main(y_target_unscaled, y_predict_unscaled))
            rmse = np.sqrt(mse)

            metrics = {'mse': mse, 'rmse': rmse}

            print(f"Mean Squared Error (MSE): {mse}")
            print(f"Root Mean Squared Error (MSE): {rmse}")

        elif self.problem_type in ['binary_class', 'multi_class']:
            correct_predictions = 0
            predicted_classes = []
            true_classes = []

            for pred, true in zip(all_y_predict, all_y_target):
                y_predict = None
                y_target = None

                if self.problem_type == 'binary_class':
                    # Apply threshold
                    y_predict = 1 if pred[0] > 0.5 else 0
                    y_target = true[0]
                else:
                    # Predicted class == highest probability
                    y_predict = np.argmax(pred)
                    y_target = np.argmax(true)

                if y_predict == y_target:
                    correct_predictions += 1

                predicted_classes.append(y_predict)
                true_classes.append(y_target)

            accuracy = (correct_predictions / n) * 100 if n > 0 else 0

            metrics = {
                'accuracy': accuracy,
                'total_samples': n,
                'correct_predictions': correct_predictions
            }

            print(f"Accuracy: {accuracy}%")
            print(f"Correct Predictions: {correct_predictions}/{n}")

        return metrics

    # - Apply a dot product between layers arrays and their respective weight matrix (wT).
    # - Apply activation function to each of the dot product results, to make it non-linear (if activation != Linear);
    # - Repeat this process until it gets the output layer outputs;
    def __forward_pass(self, inputs: list[float]) -> list[float]:
        # Reset state for the current forward pass
        self.x = []
        self.z = []

        y_predict: list[float] = inputs

        for i, weights in enumerate(self.weights):
            y_predict = [1.0] + y_predict  # Add 1 for the bias
            self.x.append(y_predict)

            v: list[float] = np.dot(weights, y_predict).tolist()
            self.z.append(v)

            activation_function = self.layers[i].activation_function
            y_predict = activation_function(v)

        return y_predict

    # - Calculate the output layer error gradient given y predict and y target, to get the y value
    # in which the error most increases.
    def __calculate_error(
            self,
            y_predict: list[float],
            y_target: list[float],
    ) -> list[float]:
        error: list[float] = []

        match self.problem_type:
            case 'regression':
                error = Loss.SquareError.main(
                    y_predict=y_predict[0],
                    y_target=y_target[0],
                )
            case 'binary_class':
                error = Loss.BinaryCrossEntropy.main(
                    y_predict=y_predict[0],
                    y_target=y_target[0],
                )
            case 'multi_class':
                error = Loss.MultiClassCrossEntropy.main(
                    y_predict=y_predict,
                    y_target=y_target,
                )

        return error

    # - Calculate the output layer error gradient given y predict and y target, to get the y value
    # in which the error most increases.
    def __calculate_error_gradient(
            self,
            y_predict: list[float],
            y_target: list[float],
    ) -> list[float]:
        d_error: list[float] = []

        match self.problem_type:
            case 'regression':
                d_error = Loss.SquareError.derivative(
                    y_predict=y_predict[0],
                    y_target=y_target[0],
                )
            case 'binary_class':
                d_error = Loss.BinaryCrossEntropy.derivative(
                    y_predict=y_predict[0],
                    y_target=y_target[0],
                )
            case 'multi_class':
                d_error = Loss.MultiClassCrossEntropy.derivative(
                    y_predict=y_predict,
                    y_target=y_target,
                )

        return d_error

    # - Backpropagate the y error gradients to previous layers
    # - Calculate gradients for the bias and weights
    def __back_propagation(self, d_y: list[float]):
        d_y_linear: list[float] = d_y

        for i in reversed(range(len(self.weights))):
            # Activation layer
            if i == len(self.weights) - 1 and self.problem_type != 'regression':
                d_y_activ = d_y_linear
            else:
                derivative_activation_function = (
                    self.layers[len(self.layers) - 1 - i].derivative_activation_function
                )

                x_derivatives: list[float] = derivative_activation_function(self.z[i])

                if x_derivatives and isinstance(x_derivatives[0], list):
                    # If it's a matrix (from softmax), need to use dot product
                    d_y_activ = np.dot(d_y_linear, x_derivatives).tolist()
                else:
                    # If it's a vector, use element-wise multiplication
                    d_y_activ: list[float] = [d_y_i * last_x_d for d_y_i, last_x_d in zip(d_y_linear, x_derivatives)]

            # Save weight error gradients for later update
            d_w: list[list[float]] = np.transpose(np.outer(self.x[i], d_y_activ)).tolist()
            self.d_weights.insert(0, d_w)

            # Transpose weights (without bias) for back-propagation
            layer_weights_without_bias: list[list[float]] = [w[1:] for w in self.weights[i]]
            w_t: list[list[float]] = np.transpose(layer_weights_without_bias).tolist()

            # Calculate the previous layer error gradient to pass to the previous layer
            d_y_linear: list[float] = np.dot(w_t, d_y_activ).tolist()

    def __update_weights(self):
        for k in range(len(self.weights)):
            w = np.array(self.weights[k])
            d_w = np.array(self.d_weights[k])

            self.weights[k] = (w - self.learning_rate * d_w).tolist()
