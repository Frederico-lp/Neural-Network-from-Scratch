import numpy as np
import math

def categorical_cross_entropy(outputs, target_outputs):
    # avoid infinite values
    clipped_target_outputs = np.clip(outputs, 1e-7, 1 - 1e-7)

    if(len(target_outputs.shape) == 1):
        # scalar values
        correct_target_outputs = clipped_target_outputs[range(len(outputs)), target_outputs]
    else:
        # one-hot enconded
        correct_target_outputs = np.sum(clipped_target_outputs * target_outputs, axis=1)

    return -np.log(correct_target_outputs)