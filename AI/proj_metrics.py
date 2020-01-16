from __future__ import division, print_function
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')

def mse(y_true, y_pred, smooth=1.0):
    # Flatten all the arrays
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return tf.squared_difference(y_true_f, y_pred_f)
