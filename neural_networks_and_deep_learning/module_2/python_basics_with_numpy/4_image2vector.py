# GRADED FUNCTION:image2vector

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    # (≈ 1 line of code)
    # v =
    # YOUR CODE STARTS HERE
    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    
    # YOUR CODE ENDS HERE
    
    return v