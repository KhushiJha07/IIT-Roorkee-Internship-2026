# Week 2: IMP CNN Architectures & Vision Transformers

## Overview
This week’s research focuses on the progression of Convolutional Neural Networks (CNNs), analyzing key architectures from LeNet to DenseNet to understand structural improvements. It also covers the inherent drawbacks of CNNs and how Vision Transformers (ViTs) utilize attention mechanisms for large-scale image recognition.

## CNN Architectures: A Comparative Analysis

| Architecture | Key Features | Parameters |
| :--- | :--- | :--- |
| **LeNet-5** | Designed for handwritten datasets (MNIST), utilizes Tanh or sigmoid activations, and employs average pooling. | ~60,000. |
| **AlexNet** | Built for 1000 categories of real-world objects (ImageNet), introduces ReLU activation, and uses max pooling. It suffers from memory inefficiency during training[cite: 1]. | ~60 million. |
| **VGGNet-16** | Uses uniform $3 \times 3$ filters instead of mixed sizes, allowing for deeper networks while managing parameter counts. | ~138 million. |
| **ResNet** | Introduces skip connections to bypass layers, allowing the network to learn the residual $f(x)$ for an output of $F(x) + x$. This solves the vanishing gradient problem where shallow layers stop learning. | ~126 million (152-layer). |
| **DenseNet** | Features dense connections where every layer is connected via concatenation, maximizing feature reuse and keeping parameter counts low. | Low (growth rate $K$ produces only 32 filters). |
While, applying this MNIST Datatset:
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/f1697e4e-06ef-4c9c-ad3b-15f1b828bd95" />

## Limitations of Traditional CNNs
*   **Context and Flexibility:** CNNs struggle with capturing global context due to locality and translation invariance constraints.
*   **Scaling Issues:** Stacking more layers to increase the receptive field often leads to vanishing gradients.

## The Shift to Vision Transformers (ViTs)
While CNNs perform well on smaller datasets, ViTs excel with massive datasets by dividing images into sequence patches.

### Core ViT Components
*   **Patch Embedding:** The original image is flattened into smaller patches (e.g., dividing a $224 \times 224$ image into 196 $16 \times 16$ patches) to make the data manageable as vectors.
*   **Positional Encoding:** Values are assigned to patches to retain spatial location information, which would otherwise be isolated.
*   **Self-Attention Mechanism:** Determines the importance of different patches relative to each other using Query (Q), Key (K), and Value (V). This relationship is mathematically defined as:
    $$Attention(Q, K, V) = Softmax(\frac{QK^T}{\sqrt{d_K}})$$.
*   **Gradient Stability:** Scaling by $\sqrt{d_K}$ prevents the dot product from resulting in vanishing gradients in high dimensions. Unlike CNNs, this attention mechanism captures global context from the very first layer.
