```
Author: Cfir Hadar

Tags: Done
```
# Lesson 02 - Optimization

## Motivation

In the previous lesson we defined a neural network and a loss. What remains is the actual learning: finding weights $\theta$ that minimize the empirical loss

$$
\mathcal{L}(\theta)=\frac{1}{n}\sum_{i=1}^{n}\ell\left(f_\theta(x_i),y_i\right).
$$

For neural networks $\mathcal{L}$ is non-convex, high-dimensional, and expensive to evaluate on the full dataset — so essentially all deep learning is trained with variants of one algorithm: stochastic gradient descent, with gradients computed by backpropagation.

## Gradient Descent and Its Stochastic Version

The gradient $\nabla_\theta\mathcal{L}$ points in the direction of steepest ascent, so we step against it:

$$
\theta_{t+1}=\theta_t-\eta\,\nabla_\theta\mathcal{L}(\theta_t),
$$

with learning rate $\eta$. Computing $\nabla\mathcal{L}$ over all $n$ samples per step is wasteful; **SGD** instead estimates the gradient on a random *mini-batch* $B_t$:

$$
\theta_{t+1}=\theta_t-\eta\,\frac{1}{|B_t|}\sum_{i\in B_t}\nabla_\theta\,\ell\left(f_\theta(x_i),y_i\right).
$$

The mini-batch gradient is an *unbiased* estimator of the full gradient; its noise scales like $1/\sqrt{|B_t|}$. The noise is not purely a nuisance — it helps escape saddle points and sharp minima, and is part of why SGD generalizes well.

The learning rate is the single most important hyperparameter: too large diverges, too small crawls and gets stuck. In practice we *schedule* it — warm up briefly, then decay (step, cosine, etc.).

## Backpropagation

Backpropagation is nothing more than the chain rule, organized so that each intermediate quantity is computed exactly once. Write the network as a composition $f=f_L\circ\cdots\circ f_1$ with intermediate activations $a_l=f_l(a_{l-1})$. The loss gradient flows backward:

$$
\frac{\partial \mathcal{L}}{\partial a_{l-1}}
=\left(\frac{\partial a_l}{\partial a_{l-1}}\right)^T\frac{\partial \mathcal{L}}{\partial a_l},
\qquad
\frac{\partial \mathcal{L}}{\partial \theta_l}
=\left(\frac{\partial a_l}{\partial \theta_l}\right)^T\frac{\partial \mathcal{L}}{\partial a_l}.
$$

Concretely for a fully connected layer $a_l=\phi(z_l)$, $z_l=W_l a_{l-1}+b_l$, defining the "error signal" $\delta_l=\frac{\partial\mathcal{L}}{\partial z_l}$:

$$
\delta_L=\nabla_{a}\ell\odot\phi'(z_L),\qquad
\delta_l=\left(W_{l+1}^T\delta_{l+1}\right)\odot\phi'(z_l),\qquad
\frac{\partial\mathcal{L}}{\partial W_l}=\delta_l\,a_{l-1}^T,\quad
\frac{\partial\mathcal{L}}{\partial b_l}=\delta_l .
$$

One forward pass stores the activations; one backward pass computes all gradients — total cost about **twice** a forward pass, independent of the number of parameters. This efficiency (reverse-mode automatic differentiation) is what makes training billion-parameter models possible; frameworks like PyTorch do it for you (`loss.backward()`), building the computation graph as you execute the forward pass.

The recursion $\delta_l=(W_{l+1}^T\delta_{l+1})\odot\phi'(z_l)$ also explains the classic pathologies: a product of many factors smaller than 1 → **vanishing gradients** (early layers stop learning), larger than 1 → **exploding gradients**. Remedies you have already met or will meet: ReLU-family activations ($\phi'=1$ on the active half), careful initialization (variance-preserving, e.g. He/Xavier), normalization layers, residual connections, and gradient clipping (standard for RNNs).

## Momentum

Plain SGD oscillates across steep directions and crawls along flat ones. **Momentum** accumulates an exponential moving average of gradients:

$$
v_{t+1}=\beta v_t+\nabla\mathcal{L}(\theta_t),\qquad
\theta_{t+1}=\theta_t-\eta\, v_{t+1},
$$

with $\beta\approx0.9$. Think of a heavy ball rolling on the loss surface: oscillating gradient components cancel in the average, consistent ones accumulate — faster and smoother progress in ravines.

## Adaptive Methods: RMSProp and Adam

Different parameters can need very different step sizes. RMSProp normalizes each coordinate by a running average of its squared gradient. **Adam** combines this with momentum:

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2,
$$

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},\quad
\hat v_t=\frac{v_t}{1-\beta_2^t},\qquad
\theta_{t+1}=\theta_t-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon},
$$

with defaults $\beta_1=0.9$, $\beta_2=0.999$. The $\hat{}$ corrections remove the startup bias of the zero-initialized averages. Each parameter gets an effective step size $\eta/\sqrt{\hat v_t}$: rarely/weakly updated parameters take larger steps. **AdamW** decouples weight decay from the gradient update (apply $\theta\leftarrow(1-\eta\lambda)\theta$ separately) and is the standard optimizer for transformers.

Practical default: AdamW with a warmup + cosine-decay schedule. Plain SGD+momentum remains competitive (sometimes better generalization) for convnets, at the price of more learning-rate tuning.

## The Full Training Loop

Everything above assembles into the loop you will write in every project:

```python
for epoch in range(num_epochs):
    for x, y in dataloader:          # mini-batches
        optimizer.zero_grad()        # clear old gradients
        loss = criterion(model(x), y)  # forward pass
        loss.backward()              # backprop: compute gradients
        optimizer.step()             # update weights
```

Monitor the *training* loss to verify optimization works, and a *validation* loss to know when to stop (early stopping is regularization, not a compromise).

## Assignment

Do-it-yourself backpropagation: take a 2-layer network ($x\in\mathbb{R}^2$, one hidden layer of width 2 with sigmoid, scalar output, squared loss), pick concrete numeric weights, and compute one full forward and backward pass **by hand**, then one SGD update with $\eta=0.1$. Verify your numbers against PyTorch by comparing to `loss.backward()` gradients.

## Walkthrough

The training loops inside [FCNN on MNIST](../walkthroughs/lesson1_fcnn_mnist.ipynb) and [CNN on CIFAR-10](../walkthroughs/lesson3_cnn_cifar10.ipynb) are this lesson in code — revisit them and swap optimizers/schedules to see the effects.
