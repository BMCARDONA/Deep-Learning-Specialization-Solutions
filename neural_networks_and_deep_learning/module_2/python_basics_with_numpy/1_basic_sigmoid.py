import math
from public_tests import *

# GRADED FUNCTION: basic_sigmoid

def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """
    # (≈ 1 line of code)
    # s = 
    
    # YOUR CODE STARTS HERE
    s = 1 / (1 + np.exp(-x))
    # YOUR CODE ENDS HERE
    
    return s