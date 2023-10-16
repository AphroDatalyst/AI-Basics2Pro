Deep learning training involves many hyperparameters, and some of them might seem similar or interconnected but serve different purposes. Here are a few hyperparameters and concepts that sometimes get confused:

### 1. **Batch Size vs Mini-batch Size**:
- **Batch Size**: Refers to the number of training examples utilized in one iteration. In the context of deep learning and when using the entire dataset at once, it's called "batch gradient descent."
  
- **Mini-batch Size**: In practice, using the entire dataset can be computationally expensive, so the dataset is divided into smaller batches called "mini-batches." When using mini-batches, the training algorithm is often termed "mini-batch gradient descent" or just "batch gradient descent."

### 2. **Momentum vs Nesterov Momentum**:
- **Momentum**: It's like a moving average of the gradients, helping the parameter updates to be in consistent directions, which can help in faster convergence.
  
- **Nesterov Momentum (Nesterov Accelerated Gradient)**: A variation of the momentum method which computes the gradient at the anticipated position in parameter space (after the current momentum step). It often converges faster or more stably than standard momentum.

### 3. **Dropout vs DropConnect**:
- **Dropout**: A regularization technique where, during training, randomly selected neurons are ignored, meaning their weights aren't updated.
  
- **DropConnect**: A generalization of dropout where individual weights (rather than neurons) are set to zero during training.

### 4. **Activation Functions: Sigmoid vs Tanh vs ReLU**:
- **Sigmoid**: An activation function that squashes values between 0 and 1. It can suffer from vanishing gradient problems.
  
- **Tanh (Hyperbolic Tangent)**: Similar to the sigmoid but squashes values between -1 and 1. It's zero-centered, which helps mitigate some of the issues of the sigmoid.
  
- **ReLU (Rectified Linear Unit)**: An activation function that outputs the input directly if positive, otherwise, it outputs zero. It's widely used because of its computational efficiency and tendency to converge faster.

### 5. **Early Stopping vs Patience**:
- **Early Stopping**: A regularization method where training is stopped as soon as the validation performance starts degrading, preventing potential overfitting.
  
- **Patience**: Often used in conjunction with early stopping. It's the number of epochs to wait before stopping once the validation performance starts degrading. It ensures that training isn't stopped just due to minor fluctuations in validation metrics.

These are just a few examples. The deep learning field is vast, with many techniques and hyperparameters, each designed to address specific challenges or enhance model performance.