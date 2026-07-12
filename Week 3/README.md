# Week 3: Deep Dive into Vision Transformers (ViT)

## Research Summary
This week involved a comprehensive analysis of the landmark paper: *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale."* My focus was on understanding the structural departure from Convolutional Neural Networks (CNNs) and the mathematical foundations of the self-attention mechanism in computer vision.

## Key Technical Insights
*   **Inductive Bias vs. Global Context:** Unlike CNNs, which have a strong inductive bias (locality/translation invariance), ViTs have minimal spatial priors, allowing them to learn global relationships from the very first layer given sufficient data.
*   **The Patching Pipeline:** Images are reshaped from $H \times W \times C$ into a sequence of flattened 2D patches $x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}$, where $N$ is the number of patches.
*   **Tokenization Components:** 
    *   **Linear Projection:** Mapping patches into a constant latent dimension $D$.
    *   **CLS Token:** A learnable embedding used as the final image summary for the classification head.
    *   **Positional Embeddings:** Learnable 1D vectors added to patch embeddings to retain spatial grid information.

## State-of-the-Art Exploration: XAAI-ledger
I explored hybrid architectures, specifically **XAAI-ledger**, a multi-modal framework for melanoma detection. It integrates dermoscopic imagery and clinical metadata using a CNN-Transformer encoder and attention-based fusion, achieving early-stage diagnostic transparency through Grad-CAM and SHAP.


