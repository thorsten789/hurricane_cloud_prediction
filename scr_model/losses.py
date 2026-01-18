import tensorflow as tf

@tf.function
def cubic_mse(y_true, y_pred):
    """
    Cubic Mean Absolute Error:
    mean(|y_true - y_pred|^3)
    """
    error = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.pow(error, 3))