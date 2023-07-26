# GRADED FUNCTION: schedule_lr_decay

def schedule_lr_decay(learning_rate0, epoch_num, decay_rate, time_interval=1000):
    """
    Calculates updated the learning rate using exponential weight decay.
    
    Arguments:
    learning_rate0 -- Original learning rate. Scalar
    epoch_num -- Epoch number. Integer.
    decay_rate -- Decay rate. Scalar.
    time_interval -- Number of epochs where you update the learning rate.

    Returns:
    learning_rate -- Updated learning rate. Scalar 
    """
    # (approx. 1 lines)
    # learning_rate = ...
    
    # YOUR CODE STARTS HERE
    # If epoch_num = 3,174 and timeInterval = 1000, then math.floor(epoch_num / time_interval) = 3
    # If epoch_num = 5,062 and timeInterval = 1000, then math.floor(epoch_num / time_interval) = 5
    # The denominator of the decay function increases over time, which thereby decreases the value of 
    # the learning rate. 
    learning_rate = learning_rate0 / (1 + decay_rate * math.floor(epoch_num / time_interval))
    # YOUR CODE ENDS HERE
    
    return learning_rate