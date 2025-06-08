import numpy as np

from main import input_layer_size, hidden_layer_size, output_layer_size


class NeuralNetwork:
    def __init__(
            self,
            problem_type: int,
            input_layer_size: int,
            hidden_layers_sizes: list[int],
            output_layer_size: int,
            activation: int,
            loss: int,
    ):
        self.problem_type = problem_type
        self.input_layer_size = input_layer_size
        self.hidden_layers_sizes = hidden_layers_sizes
        self.output_layer_size = output_layer_size
        self.activation = activation
        self.loss = loss

    # - Apply dot product (produto escalar) between input array (column size 1, row size "input_layer_size") and weight
    # matrix (wT) (row size "input_layer_size", column size "hidden_layers_sizes[i]" or "output_layer_size"). Each weight is a connection
    # between an input neuron and a hidden (or output, if zero hidden layers) neuron. The input array starts with a 1
    # for the Bias b. The dot product will output an array of linear float values (column size 1, row size "hidden_layers_sizes[i]" or "output_layer_size").

    # - Apply activation function to each of the dot product results, to make it non-linear (if activation != Linear);
    # - Repeat this process until it gets the output layer outputs;
    def forward(self, inputs: list[float]) -> list[float]:
        v: list[float] = np.dot(inputToHidden1InitialWeights, inputs)

        # y = sigmoid(v)

        return v  # y

    # - Calculate loss based on the predicted y and correct y
    # - Calculate gradient descent to get the direction (value) where the error decreases
    # - Calculate the new bias and weights for next iteration
    # - Backpropagate the weights
    def backpropagation(self, y_predict, y_target):
        ...

    # Run all together
    def train(self, X, Y, epochs, lr):
        ...


# activation_to_loss = {
#     Activation.Linear.value: Loss.Mse.value,
#     Activation.Sigmoid.value: Loss.CrossEntropy.value,
#     Activation.Softmax.value: Loss.CrossEntropy.value,
# }

inputToHidden1InitialWeights: list[list[float]] = (
    np.random.uniform(low=1, high=101, size=(hidden_layer_size, input_layer_size)).tolist()
)
hidden1ToOutputInitialWeights: list[list[float]] = (
    np.random.uniform(low=1, high=101, size=(output_layer_size, hidden_layer_size)).tolist()
)

# inputToHidden1InitialWeights:
# [b1, w11, w12, w13, w14]
# [b2, w21, w22, w23, w24]
# [b3, w31, w32, w33, w34]
# [b4, w41, w42, w43, w44]
