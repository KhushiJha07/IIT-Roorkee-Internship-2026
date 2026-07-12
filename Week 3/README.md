# Week 3: Vision Transformers (ViT)

This week involved a comprehensive analysis of the landmark paper: *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale."* My focus was on understanding the structural departure from Convolutional Neural Networks (CNNs) and the mathematical foundations of the self-attention mechanism in computer vision. 

<img width="1418" height="808" alt="image" src="https://github.com/user-attachments/assets/9840e1e5-df8e-4adb-af95-e2d19efc8b35" />

## Key Technical Insights
*   **Inductive Bias vs. Global Context:** Unlike CNNs, which have a strong inductive bias (locality/translation invariance), ViTs have minimal spatial priors, allowing them to learn global relationships from the very first layer given sufficient data.
*   **The Patching Pipeline:** Images are reshaped from $H \times W \times C$ into a sequence of flattened 2D patches $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where $N$ is the number of patches.
*   **Tokenization Components:** 
    *   **Linear Projection:** Mapping patches into a constant latent dimension $D$.
    *   **CLS Token:** A learnable embedding used as the final image summary for the classification head.
    *   **Positional Embeddings:** Learnable 1D vectors added to patch embeddings to retain spatial grid information.
      
## Architectural Comparison: CNNs vs. ViTs

To better understand the structural departure from traditional models, I compared the fundamental mechanics of Convolutional Neural Networks against Vision Transformers.


### Comparative Analysis Table

| Architectural Feature | Convolutional Neural Networks (CNNs) | Vision Transformers (ViTs) |
| :--- | :--- | :--- |
| **Core Mechanism** | Convolutions and Pooling | Patching and Self-Attention Mechanism |
| **Inductive Bias** | **High:** Assumes locality and translation invariance natively. | **Low:** Minimal spatial priors; learns relationships entirely from data. |
| **Data Dependency** | Highly efficient and performs well on small-to-medium datasets. | Requires massive datasets (e.g., JFT-300M) to establish spatial priors and outperform CNNs. |
| **Receptive Field** | **Local:** Field grows gradually deeper into the network. | **Global:** Captures long-range dependencies from the very first layer. |
| **Computational Complexity** | Scales linearly with image resolution. | Scales quadratically ($O(N^2)$) with the number of patches ($N$). |
| **Feature Processing** | Hierarchical (edges $\rightarrow$ textures $\rightarrow$ shapes). | Sequence-based (treats image patches identical to text tokens). |
| **Interpretability** | Visualized via Grad-CAM or intermediate feature maps. | Visualized via Attention Maps (showing exact patch-to-patch focus). |

