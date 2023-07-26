**What you should remember**:
- Shuffling and Partitioning are the two steps required to build mini-batches
- Powers of two are often chosen to be the mini-batch size, e.g., 16, 32, 64, 128.
- Momentum takes past gradients into account to smooth out the steps of gradient descent. It can be applied with batch gradient descent, mini-batch gradient descent or stochastic gradient descent.
- You have to tune a momentum hyperparameter $\beta$ and a learning rate $\alpha$.
- The accuracy of Mini-batch GD or Mini-batch GD (with Momentum) is significantly lower than that of Adam; but when learning rate decay is added on top, either can achieve performance at a speed and accuracy score that's similar to Adam.