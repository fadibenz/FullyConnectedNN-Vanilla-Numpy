# Fully Connected Neural Network Implementation

A modular implementation of fully connected neural networks from scratch, featuring forward and backward propagation, various layer types, and a flexible solver for training. This project is based on UC Berkeley's CS182 course materials.

## Project Overview

This implementation provides a deep dive into neural network fundamentals by building everything from the ground up. Key features include:

- Modular layer architecture with forward and backward passes
- Support for networks with arbitrary hidden layers
- Comprehensive gradient checking
- SGD optimization with learning rate decay
- Various loss functions (Softmax, SVM)
- Training visualization capabilities

## Components

### Layer Implementations

- **Affine Layer**: Implements fully connected layers with weights and biases
- **ReLU Activation**: Implements the Rectified Linear Unit activation function
- **Loss Layers**: 
  - Softmax loss for classification
  - SVM loss as an alternative loss function
- **Composite Layers**: 
  - Affine-ReLU sandwich combining affine transformation with ReLU activation

### Network Architectures

1. **Two Layer Network**: A simple network with one hidden layer
2. **Fully Connected Network**: A flexible implementation supporting arbitrary depth
   - Configurable hidden layer sizes
   - Weight scale initialization
   - Regularization support

### Training Infrastructure

- **Solver Class**: Handles the training process with features like:
  - Mini-batch gradient descent
  - Learning rate decay
  - Training history tracking
  - Model checkpointing
  - Performance visualization

## Requirements

```python
numpy
matplotlib
```

## Usage

### Basic Network Creation

```python
# Create a two-layer network
model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=1e-2)

# Create a deeper network
model = FullyConnectedNet([H1, H2, H3], input_dim=D, num_classes=C,
                         weight_scale=1e-2, reg=0.0)
```

### Training

```python
solver = Solver(model, data,
                optim_config={'learning_rate': 1e-3},
                lr_decay=0.95,
                num_epochs=10,
                batch_size=100,
                print_every=100)
solver.train()
```

### Visualization

```python
# Plot training loss
plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.plot(solver.loss_history, 'o')
plt.xlabel('Iteration')

# Plot classification accuracy
plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.plot(solver.train_acc_history, '-o', label='train')
plt.plot(solver.val_acc_history, '-o', label='val')
plt.legend(loc='lower right')
```

## Implementation Details

### Layer Forward/Backward API

Each layer implements:
- `forward(x)`: Takes input and returns output and cache
- `backward(dout)`: Takes gradient of loss and cache, returns gradient with respect to input

### Weight Initialization

- Uses He initialization by default
- Configurable weight scale parameter
- Supports various initialization strategies

### Training Features

- Mini-batch sampling
- Learning rate scheduling
- Loss and accuracy history tracking
- Model checkpointing
- Performance visualization

## Results

The implementation successfully demonstrates:
- Ability to overfit small datasets (50 samples) to 100% accuracy
- Stable training with larger datasets
- Proper gradient flow through deep architectures
- Expected behavior with different regularization strengths

## Development Notes

1. Current implementation focuses on core neural network components
2. Future additions could include:
   - Dropout regularization
   - Batch normalization
   - Additional optimization algorithms
   - More activation functions

## File Structure

```
deeplearning/
├── classifiers/
│   ├── fc_net.py        
├── layers.py           # Individual layer implementations
├──layer_utils.py       # Network implementation # Composite layer utilities
├── solver.py           # Training infrastructure
├── gradient_check.py   # Gradient checking utilities
└── data_utils.py       # Data loading and preprocessing
```

## Acknowledgments

This implementation is based on UC Berkeley's CS182 course materials and serves as an educational tool for understanding neural network fundamentals.