import numpy as np

np.random.seed(1)
alpha = 0.1
hidden_size = 4

streetlights = np.array([
    [1, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
])

walk_vs_stop = np.array([[1, 1, 0, 0]]).T


def relu(x):
    """корректировка веса в трехслойной сети, для поиска корреляуии, если вес меньше нуля то привести его к нулю"""
    return (x < 0) * x


weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
weights_1_2 = 2 * np.random.random((3, hidden_size)) - 1

layer_0 = streetlights[0]
layer_1 = relu(np.dot(layer_0, weights_0_1))
layer_2 = relu(np.dot(layer_1, weights_1_2))
#print("weights_0_1", str(weights_0_1))
