import tensorflow as tf

@tf.function
def cubic_mse(y_true, y_pred):
    """
    Cubic Mean Absolute Error:
    mean(|y_true - y_pred|^3)
    """
    error = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.pow(error, 3))

@tf.function
def cubic_mse_masked(y_true, y_pred, mask_value=-999.0):
    mask = tf.not_equal(y_true, mask_value)           # True, wo y_true != -999
    mask = tf.cast(mask, tf.float32)
    
    error = tf.abs(y_true - y_pred) ** 3
    error = error * mask                              # maskierte Werte = 0

    denom = tf.reduce_sum(mask)
    denom = tf.maximum(denom, 1.0)   # verhindert Division durch 0
    return tf.reduce_sum(error) / tf.reduce_sum(mask) # Mittelwert nur über echte Werte