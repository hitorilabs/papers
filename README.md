# papers
Now that I have a 3090, I think it'll be an interesting
exercise to go through key papers in deep learning
history. 

In an effort to cover my bases, I have the deep learning
book with me.

This is not a repo about the contents of these papers,
instead it's a log of things I personally learned along
the way. Everything will be done in `torch`.

![](https://github.com/hitorilabs/papers/assets/131238467/52a1e456-dd13-402a-a2ce-3c8fb35105cb)
*Deep Learning (Ian J. Goodfellow, Yoshua Bengio and Aaron Courville), MIT Press, 2016.*

## Setup

```bash
pip install torch numpy pandas
```

- cybernetics + "model of a neuron" (McCulloch and Pitts, 1943; Hebb, 1949)
- perceptron (Rosenblatt, 1958)
- adaptive linear element (ADALINE)
- back-propagation (Rumelhart et al., 1986)
- deep learning (Hinton et al., 2006; Bengio et al., 2007; Ranzato et al., 2007)

Important Papers:
- Paper linked in [PyTorch SGD Implementation](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html) - Nesterov momentum [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf)
- AdamW (the goal is not exactly convergence)
- Early Stopping https://github.com/Bjarten/early-stopping-pytorch# 

# 1958 - perceptron + adaline

The usecase was mostly for simple binary classifiers. To
demonstrate the perceptron + adaline:
- use a single linear layer
- zero initialized weights + biases
- stochastic gradient descent (SGD) optimizer
- mean squared error (MSE) loss

This is probably the simplest form of backpropagation

Deep learning was heavily inspired by the brain, but most
advancements were made by engineering.

- 1975 - 1980 introduced the neocognitron
- 1986 - connectionism / parallel distributed processing https://stanford.edu/~jlmcc/papers/PDP/Chapter1.pdf
- distributed representation https://web.stanford.edu/~jlmcc/papers/PDP/Chapter3.pdf

> This is the idea that each input to a system should be represented by many features, and each feature should be involved in the representation of many possible inputs

1990s progress in modeling sequences with neural networks. 

- Hochreiter (1991) and Bengio et al. (1994) identified some of thge fundamental mathematical difficulties in modeling long sequences.
- Hochreiter and Schmidhuber (1997) introduced long short-term memory (LSTM) network to resolve some difficulties.
- Kernel machines (Boser et al., 199; Cortes and Vapnik, 1995; Scholkopf et al., 1999) and graphical models (Jordan, 1998) achieved good results on many important tasks. (led to a decline in popularity with neural networks)
- Canadian Institute for Advanced Research (CIFAR) played a key role in keeping neural network research alive. This united machine learning groups led by Geoffrey Hinton, Yoshua Bengio, Yann LeCun.

1. Perceptron (Rosenblatt, 1958, 1962)
2. Adaptive linear element (Widrow and Hoﬀ, 1960)
3. Neocognitron (Fukushima, 1980)
4. Early back-propagation network (Rumelhart et al., 1986b)
5. Recurrent neural network for speech recognition (Robinson and Fallside, 1991)
6. Multilayer perceptron for speech recognition (Bengio et al., 1991)
7. Mean ﬁeld sigmoid belief network (Saul et al., 1996)
8. LeNet-5 (LeCun et al., 1998b)
9. Echo state network (Jaeger and Haas, 2004)
10. Deep belief network (Hinton et al., 2006)
11. GPU-accelerated convolutional network (Chellapilla et al., 2006)
12. Deep Boltzmann machine (Salakhutdinov and Hinton, 2009a)
13. GPU-accelerated deep belief network (Raina et al., 2009)
14. Unsupervised convolutional network (Jarrett et al., 2009)
15. GPU-accelerated multilayer perceptron (Ciresan et al., 2010)
16. OMP-1 network (Coates and Ng, 2011)
17. Distributed autoencoder (Le et al., 2012)
18. Multi-GPU convolutional network (Krizhevsky et al., 2012)
19. COTS HPC unsupervised convolutional network (Coates et al., 2013)
20. GoogLeNet (Szegedy et al., 2014a)

LSTMs were thought to revolutionize machine translation
(Sutskever et al., 2014; Bahdanau et al., 2015) when the
book was published back in 2016