Rosenblatt’s perceptron is built around a nonlinear
neuron, namely, the McCulloch–Pitts model of a neuron. 

The perceptron + adaline first introduced the idea that we
could learn simple linear functions based on data. It was
able to achieve impressive results (at the time) for
solving binary classification problems.

The iris dataset is a classic starting point for anyone
starting off with classification tasks because it has only
4 features and 3 output classes.

https://archive.ics.uci.edu/ml/machine-learning-databases/iris/

In my implementation of adaline, I modified the dataset so
that one of the classes is excluded as to turn the problem
back into a binary classification task.

For the sake of comparison, I've also included
implementations for MLP and multiclass classification

- Data Normalization
- Initialization for weights and biases
- Minibatch vs. full batch
