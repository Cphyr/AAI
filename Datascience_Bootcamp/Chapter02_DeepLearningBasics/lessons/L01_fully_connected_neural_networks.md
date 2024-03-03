# Fully Connected Neural Networks

## Multi-Layer Perceptron (MLP)

The connection between data science and neurons' structure in our brain is merely. Yet, MLP, which is the basic building block of every neural network, is based on a single neural functioning in our brain. Neurons receive multiple inputs, sum them with a bias, and pass it through an activation function (usually non-linear function).

<img src="https://miro.medium.com/v2/resize:fit:1358/1*qQPpdtR0r1APiEfTqN74aA.png" alt="The Basics of Neural Networks (Neural Network Series) — Part 1 | by Kiprono  Elijah Koech | Towards Data Science" style="zoom:50%;" />

MLP recieves an input vector $Fx\in\mathbb{R}^d$, which is them multiply by a weights vector  $w\in\mathbb{R}^d$, and sum them with a bias $b\in\mathbb{R}$, $\sum_{i=1}^d{w_i\cdot x_i} + b$ or in matrix formation: $\langle w, x \rangle + b$. This weighted sum is then passed through an activation function, $g$ (usually denotated by $\phi$) to form the output $\hat{y} = g(\langle w,x \rangle + b)$.

Specifically were one to choose the sigmoid actiovation function $\sigma$, the result is equivalent to the basic logisitic regression.

## Linear Layer

Let us think of performing multiple logistic regressions in parallel, each with it's own weights vector and bias, and finally applying the activation function on separately each regression.

<img src="https://www.cs.rice.edu/~vo9/vislang/2017/notebooks/linear_layer.png" alt="deep_learning_lab" style="zoom:50%;" />

Mathematically, $a = g\left( W^Tx + b \right)$, whereas $W\in\mathbb{R}^{d\times l}, x\in\mathbb{R}^d, b\in\mathbb{R}^l, a\in Im(g)^l$, $l$ being the number of regression, and $g(\cdot)$​​ preformed pair regression.



## Deep Fully Connected Neural Networks

<img src="https://editor.analyticsvidhya.com/uploads/50492simple_neural_network_header.jpg" alt="Neural Network 101: Definition, Types and Application" style="zoom:50%;" />

This is the final form of our lecture, indefinitely concatenating linear layers, with nonlinear activations after each layer.

### Justifing the Nonlinear Activation

Suppose a dual layer architecture with parameters $W_1\in\mathbb{R}^{d_1\times d_2}, b_1\in\mathbb{R}^{d_2}, W_2\in\mathbb{R}^{d_2\times d_3}, b_2\in\mathbb{R}^{d_3}$ and assume that for all layers, $g\left( W_l^Tx + b_l \right) = W_l^Tx + b_l$. Hence the output of the network is
$$W_2^T\left(W_1^Tx+b_1\right)+b_2=W_2^TW_1^Tx+W_2^Tb_1+b_2=\tilde{W}^Tx+\tilde{b},$$
 which is practically equivalent for a single linear layer.

## Related Material

[But what is a neural network? | Chapter 1, Deep learning - YouTube](https://www.youtube.com/watch?v=aircAruvnKk&ab_channel=3Blue1Brown)
