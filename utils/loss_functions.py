import numpy as np

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


# MSE — Mean Squared Error
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))


def mse_derivate(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size