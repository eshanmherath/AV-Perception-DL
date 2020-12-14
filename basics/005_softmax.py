import numpy as np


def softmax(scores):
    exponential = np.exp(scores)
    summed_value = sum(exponential)
    return [i * 1.0 / summed_value for i in exponential]


original_scores = [2, 1, 0]

soft_maxed = softmax(original_scores)
print(soft_maxed)
