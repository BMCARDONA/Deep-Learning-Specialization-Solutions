# GRADED FUNCTION: sigmoid

def sigmoid(z):
    
    """
    Computes the sigmoid of z
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    a -- (tf.float32) the sigmoid of z
    """
    # tf.keras.activations.sigmoid requires float16, float32, float64, complex64, or complex128.
    
    # YOUR CODE STARTS HERE
    z = tf.cast(z, tf.float32) 
    a = tf.keras.activations.sigmoid(z)
    # YOUR CODE ENDS HERE
    
    return a