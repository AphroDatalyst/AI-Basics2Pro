Deep learning training involves many hyperparameters, and some of them might seem similar or interconnected but serve different purposes. Here are a few hyperparameters and concepts that sometimes get confused:

### 1. **Batch Size vs Mini-batch Size**:
- **Batch Size**: Refers to the number of training examples utilized in one iteration. In the context of deep learning and when using the entire dataset at once, it's called "batch gradient descent."
  
- **Mini-batch Size**: In practice, using the entire dataset can be computationally expensive, so the dataset is divided into smaller batches called "mini-batches." When using mini-batches, the training algorithm is often termed "mini-batch gradient descent" or just "batch gradient descent."


1. Batch Size:
Refers to the number of training examples used to compute a single update of the model weights.
In the context of Batch Gradient Descent (the traditional form), the batch size is equal to the total number of samples in the training dataset. This means the model weights are updated once per epoch after calculating the loss on the entire training dataset.
Benefits: The direction of the gradient step is based on the entire dataset, which can provide a more accurate estimate.
Drawbacks: It can be computationally expensive and might not fit in memory for large datasets. Additionally, it may not escape local minima as effectively as other methods.
2. Mini-batch Size:
Refers to a subset of the training dataset.
When the dataset is divided into smaller batches and each of these batches is used to compute and update the model weights, the training algorithm is termed Mini-batch Gradient Descent or, often, just "Batch Gradient Descent."
Benefits: It can lead to faster convergence, can fit in memory even for large datasets, and might escape local minima due to the noise in the gradient estimation.
Drawbacks: The direction of the gradient step is based on a subset of the dataset, which can introduce noise, leading to less stable convergence.
* Note:
Stochastic Gradient Descent (SGD): It's worth noting another variation where each single training example is used to compute the gradient and update the weights. In this case, the "batch size" is 1. It's known for its high variance in updates, which can lead to faster convergence but also more oscillation in the learning process.



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