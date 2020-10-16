from __future__ import division, print_function
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')

def mse(y_true, y_pred):
    # Flatten all the arrays
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return tf.math.squared_difference(y_true_f, y_pred_f)

def only_ocean_mse(y_true, y_pred, smooth=1.0):
    eps = .00001
    # Flatten all the arrays
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    # The true values on land should be -0.5 in order for this to work
    temp = tf.math.ceil(y_true_f + eps)  # This should make the ocean values = 1 and the land values = 0
    y_pred_c = y_pred_f * temp # Make 0 the prediction outside the prostate
    # Keep only the water values
    y_true_c = y_true_f * temp

    return tf.math.squared_difference(y_true_c, y_pred_c)

