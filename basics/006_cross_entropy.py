import numpy as np


def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))


# Probabilities of each event happening
p = [0.8, 0.7, 0.1]

# This event series has a low value for cross entropy, so its likely to happen
y = [1, 1, 0]
result = cross_entropy(y, p)
print(result)

# This event series has a high value for cross entropy, so its unlikely to happen
y = [0, 0, 0]
result = cross_entropy(y, p)
print(result)
