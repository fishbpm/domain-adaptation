import keras.backend as K
import numpy as np


def get_shapes(model, frozen):
    """Gets shape and volume (#params) for each layer in the model.

    # Arguments
        model: a keras neural network model

    # Returns
        a list of shapes (tuples) and their corresponding volume (#params)
    """
    shapes = []
    for layer in model.layers[frozen:]:
        for tensor in layer.trainable_weights:
            shapes.append([K.int_shape(tensor), K.count_params(tensor)])    
    return shapes


def set_trainable_weight(model, weights, frozen):
    """Sets the weights of the model.

    # Arguments
        model: a keras neural network model
        weights: A list of Numpy arrays with shapes and types matching
            the output of `model.trainable_weights`.
    """
    tuples = []
    for layer in model.layers[frozen:]:
        num_param = len(layer.trainable_weights)
        layer_weights = weights[:num_param]
        for sw, w in zip(layer.trainable_weights, layer_weights):
            tuples.append((sw, w))
        weights = weights[num_param:]
    K.batch_set_value(tuples)


def set_weights(model, flat, frozen):
    """Sets the weights of the model.

    # Arguments
        model: a keras neural network model
        weights: A flat array of weights matching
            the output of `get_trainable_weights`.    """
    weights = []
    shapes = get_shapes(model, frozen)
    for shape in shapes:
        weights.append(flat[:shape[1]].reshape(shape[0]))
        flat = flat[shape[1]:]
        
    set_trainable_weight(model, weights, frozen)


def get_trainable_weights(model, frozen):
    """Gets the weights of the model.

    # Arguments
        model: a keras neural network model

    # Returns
        a Numpy array of weights
    """
    W_list = K.get_session().run(model.trainable_weights)
    W_flattened_list = [k.flatten() for k in W_list[frozen:]]
    W = np.concatenate(W_flattened_list)
    return W

