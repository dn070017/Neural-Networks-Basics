# Neural Networks Basics
## Introduction
In this section, we go through the basics of deep neural networks, including the usage of [Tensor](https://www.tensorflow.org/api_docs/python/tf/Tensor) and [Variable](https://www.tensorflow.org/api_docs/python/tf/Variable) data structure in Tensorflow. Then, we introduce several [activation functions](https://www.tensorflow.org/api_docs/python/tf/keras/activations) which apply non-linear transformation to the data. To train the weights in the models with given dataset, we first introduce several [objective functions](https://www.tensorflow.org/api_docs/python/tf/keras/losses) for both regression and classification. Thereafter, we go through the gradient and jacobian matrix and how to use gradient descent and several [optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers) to find the optimal weights that can minimize the objective fucntions. Finally, we introduce [Tensorflow Data API](https://www.tensorflow.org/guide/data) and use it to implement a multilayer perceptron network to predict the digits of [MNIST dataset](http://yann.lecun.com/exdb/mnist/).

&nbsp;
## Visualization
```bash
visualization/gradient_descent.py
visualization/optimizers.py
```

&nbsp;
## Table of Contents
1. [Tensors and Variables](tensors_and_variables.ipynb)
2. [Jacobian Matrix and Gradient](jacobian_matrix_and_gradient.ipynb)
3. [Activation Functions](activation_functions.ipynb)
4. [Objective Functions](objective_functions.ipynb)
5. [Gradient Descent](gradient_descent.ipynb)
6. [Optimizers](optimizers.ipynb)
7. [Tensorflow Data](tensorflow_data.ipynb)
8. [Multilayer Perceptron](multilayer_perceptron.ipynb)