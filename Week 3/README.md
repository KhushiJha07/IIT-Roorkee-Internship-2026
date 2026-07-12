# Week 3: Deep Dive into Vision Transformers (ViT)

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

