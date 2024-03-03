# Convolution Neural Networks

Untill now we discussed fully connected neural networks, that is every neuron is connected to all neurons in the previous layer, and to all neurons in the layer proceeding it. These networks are the most expressive, as this architecture don't constrain the network. "With greater power comes great responsibility" as these are computationaly expensive. Let us consider an input gray-scale image of $1024\times1024\times1$ that is fed to a fully-connected network (FCN) with 100 output categories. Even without any hidden layers, the number of parameters in this network is $1024^2\cdot100 + 100=104,857,700$.

Solution: constrain the expressiveness of the network using 'outside' information. As humans we think of image pixels as localy related, that is close pixels are directly related, while far pixels are abstractly related.

## Convolutional Layers

The basic building block of convolution neural networks (CNNs) is the convolutional layer. Covolutional layers performs a linear convolution, which is a cross-correlation computation between the given input and learnable kernels (somethings referred to as filters).
$$
y[n] = \sum_{m=1}^{K-1}{x[n-m]\cdot w[m]},
$$
whereas, $x\in\mathbb{R}^n, w\in\mathbb{R}^K$ are the input vector and weights vector respectively, and a single kernel ($w$) contains $K$ parameters ($K \ll N_{input}\times N_{output}$​).

We can extend this formulation to process images:

<img src="https://content.codecademy.com/courses/deeplearning-with-tensorflow/image-classification/stride.gif" alt="Classification: Image Classification Cheatsheet | Codecademy" style="zoom:50%;" />

Usually, we use multiple different filters (concatenating the outputs of different filters), which ideally learn different sub-tasks.

As you can see, we distilled our knowledge of image processing, such as locality, into the network architecture, resulting in a smarter model, with significantly less parameters.

<u>Example</u>: given an input image of size $6\times 6\times 3$, with two filters, each of size $2\times2\times3$ (note that the filter's last dimension always equals to the last dimension of the input image, this dimension is referred to as channels. Therefore, conoluvtion consider a local enviroment in dimensions one and two, while considering **all** input channels at once).

<img src="https://indoml.files.wordpress.com/2018/03/convolution-with-multiple-filters2.png?w=736" alt="Student Notes: Convolutional Neural Networks (CNN) Introduction – Belajar  Pembelajaran Mesin Indonesia" style="zoom:100%;" />

Again, to break linearity, we propose performing an element-wise nonlinear activation function on each output.

### Padding, Stride and Dilation

#### Padding

Clearly, convolution is a local computation, that is not well defined for the edges, resulting in a reduced dimesionality in the output image. Padding aims to solve this phenomenon as it wraps the input image with a given symbol (usually zero, or duplicating edge values).

<img src="https://aigeekprogrammer.com/wp-content/uploads/2019/12/CNN-valid-vs.-same-1.png" alt="Convolutional neural network 2: architecture " AI Geek Programmer" style="zoom:50%;" />

#### Stride

Intuitively local information is constant in close pixels, therefore, to reduce the computational complexity, one may use a bigger stride for the filter movement.

For example, convolution with padding of one, and stride of two:

![A Guide to Convolutional Neural Networks — the ELI5 way | Saturn Cloud Blog](https://saturncloud.io/images/blog/convolution-operation-with-stride-length.gif)

Note how the filter moves two pixels at a time, instead of one.

#### Dilation (התרחבות)

In order to decrease even further in the number of computations one may increase the filter dilation. For example, filter with dilation of 2.

![File:Convolution arithmetic - Dilation.gif - Wikimedia Commons](https://upload.wikimedia.org/wikipedia/commons/c/c1/Convolution_arithmetic_-_Dilation.gif)





### Computing Output Dimension

$$
O=\frac{I-K+2P}{S}+1,
$$

whereas, $I$ is the input dimension, $K$ is the kernel size, $P$ is the padding and $S$ is the stride size.

Number of output channels equals to the number of filters as discussed before.

### Reseptive Field

Receptive Field is defined as the size of the region in the input that produces the feature.

<img src="https://miro.medium.com/v2/resize:fit:1200/1*k97NVvlMkRXau-uItlq5Gw.png" alt="Receptive Field in Deep Convolutional Networks | by Reza Kalantar | Medium" style="zoom:50%;" />

See how the $5\times5$ image is compressed to a single pixel using two convolution layers, each with $3\times3$ kernel? Therefore, the receptive field of this network is $5\times5$.

### Pooling

Usually, in spectral data close elements carry similar values (e.g., close pixels usually have similar values). 