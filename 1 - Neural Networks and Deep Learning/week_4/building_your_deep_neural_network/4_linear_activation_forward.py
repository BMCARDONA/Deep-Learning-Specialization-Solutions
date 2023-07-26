# GRADED FUNCTION: linear_activation_forward

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python tuple containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        #(≈ 2 lines of code)
        # Z, linear_cache = ...
        # A, activation_cache = ...
        
        # YOUR CODE STARTS HERE
        # linear_cache contains A, W, and and b
        Z, linear_cache = linear_forward(A_prev, W, b)
        # activation_cache contains Z
        A, activation_cache = sigmoid(Z)
        # YOUR CODE ENDS HERE
    
    elif activation == "relu":
        #(≈ 2 lines of code)
        # Z, linear_cache = ...
        # A, activation_cache = ...
        
        # YOUR CODE STARTS HERE
        # linear_cache contains A, W, and and b
        Z, linear_cache = linear_forward(A_prev, W, b)
        # activation_cache contains Z
        A, activation_cache = relu(Z)
        # YOUR CODE ENDS HERE
        
    cache = (linear_cache, activation_cache)

    return A, cache