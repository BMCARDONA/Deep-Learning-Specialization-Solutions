# GRADED FUNCTION: sigmoid

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    
    # (≈ 1 line of code)
    # s = 
    # YOUR CODE STARTS HERE
    s = 1 / (1 + np.exp(-x))
    
    # YOUR CODE ENDS HERE
    
    return s