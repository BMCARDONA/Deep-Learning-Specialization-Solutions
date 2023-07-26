# GRADED FUNCTION: update_lr

def update_lr(learning_rate0, epoch_num, decay_rate):
    """
    Calculates updated the learning rate using exponential weight decay.
    
    Arguments:
    learning_rate0 -- Original learning rate. Scalar
    epoch_num -- Epoch number. Integer
    decay_rate -- Decay rate. Scalar

    Returns:
    learning_rate -- Updated learning rate. Scalar 
    """
    #(approx. 1 line)
    # learning_rate = 
    
    # YOUR CODE STARTS HERE
    learning_rate = learning_rate0 / (1 + decay_rate * epoch_num) 
    # YOUR CODE ENDS HERE
    
    return learning_rate