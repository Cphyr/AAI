
```
Author: Cfir Hadar

Tags: Done
```

# Lesson 1 - Fully Connected Neural Networks

## Motivation

Fully connected neural networks are a very useful tool . They excel in (but are not limited to) tasks such as classification and prediction and are being used across most fields for these tasks and more.

## Multi-Layer Perceptron (MLP)

As you will see, there exists a connection between data science and neuron structure in our brains. Moreover, a multi-layer perceptron (MLP), which is the basic building block of every computational neural network, is based on a single neuron in our brain. In both cases, neurons receive multiple inputs, sum them with a bias, and pass it through some activation function (usually a non-linear one).

<img src="https://miro.medium.com/v2/resize:fit:1358/1*qQPpdtR0r1APiEfTqN74aA.png" alt="The Basics of Neural Networks (Neural Network Series) — Part 1 | by Kiprono  Elijah Koech | Towards Data Science" style="zoom:50%;" />

MLP receives an input vector $Fx\in\mathbb{R}^d$, which is then multiplied by a weight vector  $w\in\mathbb{R}^d$, and summed up with a bias $b\in\mathbb{R}$, $\sum_{i=1}^d{w_i\cdot x_i} + b$ or in matrix notation: $\langle w, x \rangle + b$. This weighted sum is then passed through an activation function, $g$ (usually denoted by $\phi$) to form the output $\hat{y} = g(\langle w,x \rangle + b)$.

Choosing an activation function is part of constricting the model and different functions will perform differently on different tasks.
Common functions that can be used are:
- ReLu
- Sigmoid
- Softmax

Specifically, if you were to choose the sigmoid activation function ($\sigma$), the result will be equivalent to the basic logistic regression model.

At this point it is less important to be familiar with the ins and outs of these functions, it is just important to remember the option to choose exists.

More examples for activation functions and the characteristics of the ones listed can be found on [Wikipedia](https://en.wikipedia.org/wiki/Activation_function):


## Linear Layer

Let's think about performing multiple logistic regressions in parallel, each with its own weight vector and bias, and then applying the activation function on each of these.

<img src="https://www.cs.rice.edu/~vo9/vislang/2017/notebooks/linear_layer.png" alt="deep_learning_lab" style="zoom:50%;" />

Mathematically it can be written as: $a = g\left(W^Tx + b \right)$, whereas $W\in\mathbb{R}^{d\times l}, x\in\mathbb{R}^d, b\in\mathbb{R}^l, a\in Im(g)^l$, $l$ being the number of regression, and $g(\cdot)$​​ preformed per regression.


## Deep Fully Connected Neural Networks

<img src="https://editor.analyticsvidhya.com/uploads/50492simple_neural_network_header.jpg" alt="Neural Network 101: Definition, Types and Application" style="zoom:50%;" />

This is the final structure of our lecture, chaining multiple linear layers, with a non-linear activation function applied after each one.

### Justifying the Nonlinear Activation

There is a big importance for choosing a nonlinear activation function. By choosing a linear activation function (for example: $g(x)=x$) our multi-layered model will be equivalent to a model with a single layer. Meaning it will give a linear result, and therefore we would not be able to learn nonlinear and more complex tasks.

Mathematically: suppose a dual layer architecture with the parameters: $W_1\in\mathbb{R}^{d_1\times d_2}, b_1\in\mathbb{R}^{d_2}, W_2\in\mathbb{R}^{d_2\times d_3}, b_2\in\mathbb{R}^{d_3}$ and assume that for all layers, $g\left( W_l^Tx + b_l \right) = W_l^Tx + b_l$. 
Hence, the output of the network is:
$$W_2^T\left(W_1^Tx+b_1\right)+b_2=W_2^TW_1^Tx+W_2^Tb_1+b_2=\tilde{W}^Tx+\tilde{b},$$
 which is practically equivalent for a single linear layer.

## Related Material

[But what is a neural network? | Chapter 1, Deep learning - YouTube](https://www.youtube.com/watch?v=aircAruvnKk&ab_channel=3Blue1Brown)


