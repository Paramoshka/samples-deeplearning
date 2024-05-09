import numpy as np

np.random.seed(1)
alpha = 0.2
hidden_size = 4

streetlights = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
])

walk_vs_stop = np.array([[1, 1, 0, 0]]).T


def relu(x):
    """корректировка веса в трехслойной сети, для поиска корреляуии, если вес больше нуля то привести его к нулю"""
    return (x > 0) * x


def relu2deriv(output):
    return output > 0


weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1

for iteration in range(200):
    print("Iteration: " + str(iteration + 1))
    layer_2_err = 0
    for i in range(len(streetlights)):
        layer_0 = streetlights[i:i+1]
        layer_1 = np.dot(layer_0, weights_0_1)
        layer_1 = relu(layer_1)
        layer_2 = np.dot(layer_1, weights_1_2)  #prediction

        layer_2_delta = layer_2 - walk_vs_stop[i]
        layer_2_err += layer_2_delta ** 2

        layer_1_delta = layer_2_delta.dot(weights_1_2.T)
        layer_1_delta *= relu2deriv(layer_1)

        weights_delta_1_2 = layer_1.T.dot(layer_2_delta)
        weights_delta_0_1 = layer_0.T.dot(layer_1_delta)
        #
        weights_1_2 -= alpha * weights_delta_1_2
        weights_0_1 -= alpha * weights_delta_0_1

        print("Prediction: " + str(layer_2))
        print("err_2 : {}".format(layer_2_err))
    print("\n")
