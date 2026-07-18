## Indian Institute of Technology Roorkee

Department of Applied Mathematics and Scientific Computing

## Evaluating CNN and ViT on Skin Cancer Detection

## A Summer Internship Report

Submitted in partial fulfilment of the requirements of the Summer Internship Programme, 25 May 2026 – 17 July

2026

Supervisor

Prof. Millie Pant

Submitted by

Khushi Jha

Mentor

Mr. Shubham Joshi

Department of Applied Mathematics and Scientific Computing

Indian Institute of Technology Roorkee

July 2026


## Certificate

This is to certify that the internship report entitled “Evaluating CNN and ViT on Skin Cancer Detection”, submitted by Khushi Jha, is a record of the work carried out by her during the Summer Internship Programme (25 May 2026 – 17 July 2026) at the Department of Applied Mathematics and Scientific

Computing, Indian Institute of Technology Roorkee, under my supervision.

The work presented in this report, to the best of my knowledge, is an authentic account of the intern’s own study, implementation, and analysis, and has not been

submitted elsewhere for the award of any other degree or diploma.

Prof. Millie Pant

Supervisor

Place: Roorkee Date: July 17, 2026

Department of Applied Mathematics

and Scientific Computing

IIT Roorkee


## Acknowledgement

Seven weeks ago, “Vision Transformer” was just a term I had seen in a paper title and quietly hoped nobody would ask me to explain. Writing that sentence now, at the end of this report, is a good way to measure how much this internship actually gave me.

I am deeply grateful to my supervisor, Prof. Millie Pant, for giving me the room to learn a fairly large field from first principles instead of jumping straight to results, and for the kind of feedback that pushes you to think rather than just to fix a plot.

I would like to sincerely thank my mentor, Mr. Shubham Joshi, for the many patient conversations about attention mechanisms, broken training runs, and the occasional confusion between a tensor’s shape and my own expectations of it. His guidance shaped not just what I built, but how I approached debugging, reading papers, and being honest about what did and did not work.

I also thank the Department of Applied Mathematics and Scientific Computing, IIT Roorkee, for hosting this internship and providing the computational resources without which none of the experiments in this report would exist.

Finally, to everyone who patiently listened to me explain the difference between a CNN and a ViT far more times than was strictly necessary – thank you for the patience.

Khushi Jha


## Abstract

Skin cancer, and melanoma in particular, is one of the few cancers where the difference between an early diagnosis and a late one can be the difference between a minor procedure and a life-threatening illness. Yet the standard diagnostic tools available to a clinician – heuristics such as the ABCDE and CASH rules – still leave unaided visual inspection accuracy in the range of 60–80%. This gap is precisely where computer vision has spent the last decade trying to help, first with Convolutional Neural Networks (CNNs), and more recently with Vision Transformers (ViTs).

This report documents a seven-week summer internship focused on understand- ing, implementing, and critically evaluating this progression – from classical image processing to CNNs, to ViTs, to a hybrid CNN-Transformer multi-modal archi- tecture. The internship began with the fundamentals of computer vision (edge detection, convolution, pooling), moved through transfer learning and the Vision Transformer architecture, and then applied both DenseNet121 and ViT-B/16 as baseline classifiers on the MILK10k (ISIC 2025) dataset. In parallel, a recent multi-modal explainable-AI framework from the literature, XAAI-Ledger, was re- produced from its published methodology on the ISIC 2019 dataset, including its Grad-CAM and SHAP based explainability components.

The results tell a fairly honest and, at times, humbling story: reproduced accuracy fell well short of the numbers reported in the source paper, largely due to class imbalance and the limited compute and time available within an internship timeline – a finding that is, in itself, a useful and common lesson in applied deep learning. The report closes by proposing a concept-bottleneck extension for future work, aimed at making the model’s diagnostic reasoning traceable to clinically meaningful concepts rather than an opaque probability score.

This document is organised as a standard internship report, with Chapter 1 introducing the problem and motivation, Chapter 2 covering the theoretical back- ground required to understand every architecture used later, and the remaining chapters (methodology, experiments, results, and future work) building on this foundation.


## Contents


## List of Figures


## List of Tables


## CHAPTER 1

## Introduction

Every year, a huge number of skin lesions get looked at, squinted at, photographed, and sometimes biopsied – all to answer one deceptively simple question: is this dangerous, or not? Most of the time the answer is no. But the small fraction of times it is yes, and specifically when that “yes” is melanoma, timing matters more than almost anything else in oncology. This chapter sets up why that tim- ing problem exists, why artificial intelligence has become a serious candidate for closing the gap, and what this internship specifically set out to do about it.

## 1.1 Artificial Intelligence in Healthcare

AI in healthcare did not begin with deep learning, even though it can feel that way from the inside of 2026. The earliest serious attempt at bringing “intelligence” into clinical decision-making was MYCIN in the 1970s – a rule-based expert system that reasoned about bacterial infections using a set of hand-coded if-then rules. It worked reasonably well on paper, and essentially never left the paper, because rule systems are brittle: every new edge case needs a new rule, and medicine is mostly edge cases.

The next real shift came with statistical machine learning in the 1990s and 2000s, particularly in radiology, where Computer-Aided Diagnosis (CAD) systems started flagging suspicious regions in mammograms and chest X-rays using hand- engineered features fed into classifiers like support vector machines. These systems were an improvement, but they were only ever as good as the features a human decided to extract – and deciding which features matter for a disease that presents in a thousand different ways is, itself, an extremely hard problem.

Deep learning changed the question entirely. Instead of an expert manually de- ciding that “asymmetry” or “edge irregularity” matters, a convolutional network could be shown thousands of labelled images and left to discover which patterns ac- tually correlate with the diagnosis. The 2012 ImageNet moment (AlexNet) is usu- ally cited as the spark for the broader field, and by 2016 the same idea had reached dermatology directly, with CNN-based classifiers matching board-certified derma- tologists on curated benchmarks. Vision Transformers entered medical imaging a few years later, bringing a different inductive bias – global self-attention instead of local convolution – and the last few years have pushed toward multi-modal,


explainable systems that try to combine image evidence with patient metadata and justify their own predictions. Figure 1.1 lays this progression out on a single timeline. [URL 🔗](#page-0)

*Figure 1.1: A brief evolution of AI in healthcare, from rule-based expert systems to modern multi-modal, explainable clinical AI.*

It is worth being honest about what CAD systems actually are before we go further, because the term gets thrown around loosely. A CAD system does not make a diagnosis. It makes a suggestion – a second opinion that a clinician is free to override. This distinction matters a lot for how the rest of this report should be read: none of the models discussed here are being proposed as replacements for a dermatologist. They are being evaluated as candidates for exactly this second- opinion role.

## 1.2 Computer Vision in Medical Imaging

Medical image analysis has walked the same path that computer vision walked more broadly, just a little later and a little more cautiously (understandably – mistakes here have real consequences). That path looks roughly like Figure 1.2: it starts with traditional, hand-crafted image processing, moves into general deep learning, narrows into convolutional architectures built specifically for images, and most recently widens again into transformer-based architectures that were originally designed for language. [URL 🔗](#page-0)

*Figure 1.2: The broad evolution of computer vision techniques used in medical imaging, from hand-crafted filters to attention-based transformer architectures.*

Traditional computer vision relied on filters designed by people – edge detec- tors, blob detectors, texture descriptors – which is precisely the kind of hand- engineering that made early CAD systems brittle. Deep learning replaced hand- designed features with learned ones, CNNs specialised that idea for grid-structured


data like images, and Vision Transformers are the most recent attempt at asking whether a more general, attention-based architecture can do the same job while modelling relationships across the entire image rather than a local neighbourhood. Chapter 2 is dedicated entirely to unpacking every stage of this diagram in techni- cal detail; this section exists only to give the reader a map before we start walking through it. [URL 🔗](#page-0)

## 1.3 Skin Cancer Detection

Skin cancers broadly split into two families. Non-melanoma skin cancers – primarily basal cell carcinoma (BCC) and squamous cell carcinoma (SCC) – are common, usually slow-growing, and rarely fatal if treated. Melanoma is far rarer but disproportionately dangerous: it accounts for a small fraction of skin cancer cases but the large majority of skin-cancer deaths, because it can metastasize to other organs if left untreated.

This is exactly why early diagnosis is the single biggest lever available. Lo- calised melanoma, caught before it spreads, has a five-year survival rate above 99%. Once it has spread to distant organs, that number falls below 30%. There is no equivalent jump anywhere else in the treatment pipeline – no drug, no surgical technique – that moves survival rates this much. The entire value proposition of an AI screening tool for skin cancer rests on this one fact: if it can nudge diagnosis earlier, even slightly, it is doing something that really matters.

Clinically, dermatologists use two well-known heuristics to structure their vi- sual inspection, both referenced directly on the ISIC dataset documentation and in the wider dermatology literature:

-  ABCDE rule – Asymmetry, Border irregularity, Color variation, Diam- eter (typically > 6 mm), and Evolution (change over time) [2]. [URL 🔗](#page-0)

-  CASH rule – Color, Architecture, Symmetry, and Homogeneity, often used as a complementary framework in dermatoscopic assessment [3]. [URL 🔗](#page-0)

Figure 1.3 sketches the ABCDE criteria schematically – not as clinical pho- tographs, but as a simplified visual mnemonic, which is really all the rule is meant to be in the first place. [URL 🔗](#page-0)

The catch, and the reason this whole report exists, is that these rules depend entirely on the skill and consistency of the person applying them. Two differ- ent clinicians looking at the same lesion do not always agree, and inter-observer variability is a well-documented problem in dermatology, with even experienced clinicians’ unaided accuracy sitting around 60–80% [1]. This is the gap AI is being [URL 🔗](#page-0)


*Figure 1.3: Schematic illustration of the ABCDE rule used in manual dermato- logical screening (Asymmetry, Border irregularity, Color variation, Diameter, Evo- lution). Illustration is a simplified mnemonic diagram, not a clinical image.*

asked to fill: not to replace clinical judgement, but to offer a second, consistent opinion that does not get tired, does not have an off day, and has effectively seen more lesions than any single clinician ever will.

## 1.4 Problem Statement

Framed plainly: manual visual screening for melanoma is inconsistent, heuristic- based, and caps out at around 60–80% accuracy even among experienced clinicians [1]. Deep learning models promise a more consistent alternative, but in practice they inherit a set of problems that are easy to state and surprisingly hard to fix: [URL 🔗](#page-0)

-  Visual similarity across classes. Early melanoma and a benign nevus can look almost identical to the naked eye (and, it turns out, to a CNN too) – the differences are often subtle textural cues rather than obvious shape differences.

-  Class imbalance. Malignant cases are, thankfully, rare in the population – which is great for public health and genuinely unhelpful for training a classifier. Figure 1.4 shows exactly how lopsided this gets on the ISIC 2019 dataset used later in this report: benign cases outnumber melanoma cases by more than four to one. [URL 🔗](#page-0)

-  Limited and inconsistent data. Dermoscopic datasets vary in imaging device, lighting, patient demographics, and label quality, which makes a model trained on one dataset fragile when it meets another.

-  Metadata ignorance. A large share of published skin-cancer classifiers are image-only, even though a dermatologist would never ignore a patient’s age, sex, or the anatomical site of a lesion when forming a judgement.

-  Lack of explainability. A model that outputs “87% melanoma” with no justification is not something a clinician can act on with confidence – and regulatory approval for clinical AI increasingly requires some form of interpretability.


*Figure 1.4: Class imbalance in the ISIC 2019 dataset used in this internship: 20,809 non-melanoma samples versus 4,522 melanoma samples – a ratio of roughly 4.6:1.*

Every one of these five problems reappears later in this report, either as a design consideration in the models that were built, or as a direct explanation for why the reproduced results in later chapters fall short of the numbers reported in the literature.

## 1.5 Motivation

It would be easy to say “CNNs work well for images, so let’s use a CNN,” and honestly, that was more or less my starting assumption before this internship. A few weeks in, that assumption did not survive contact with the literature or the data.

CNNs are excellent at picking up local texture – edges, colour gradients, small- scale patterns – because that is exactly what a convolutional filter is built to do. But melanoma diagnosis often depends on relationships across the whole lesion: is the colour distribution consistent from one edge to the other, does the border look different on one side than the other. A convolution kernel, by design, only looks at a small neighbourhood at a time; it has to stack many layers before it can reason about the image as a whole, and even then, that global reasoning is implicit and diluted.

Vision Transformers were the obvious next thing to try, because self-attention lets every patch of the image attend to every other patch from the very first layer – global context, right from the start. But ViTs are notoriously data-hungry, and the datasets available in dermatology, while not tiny, are nowhere close to the scale ViTs were originally designed for. So neither architecture, taken alone, is obviously “enough.”

The piece that convinced me the field is moving in the right direction, though,


was metadata. A 45-year-old patient with a changing lesion on sun-exposed skin is a different clinical picture from a 20-year-old with a stable lesion on a rarely- exposed site – even if the two lesions look nearly identical in a photograph. Image- only models are, by construction, blind to this information. That is the real mo- tivation behind this internship: not just to benchmark CNN against ViT, but to understand why the field is increasingly moving toward architectures that fuse visual features with structured clinical metadata, and to reproduce one such ar- chitecture end-to-end to see, honestly, how well it holds up outside its original paper.

## 1.6 Objectives of the Internship

Rather than framing this as a research contribution with novel objectives, it is more honest to frame it the way it actually happened – as a structured learning path with concrete checkpoints:

- 1. Build a solid working understanding of classical and deep-learning-based computer vision, starting from edge detection up to convolutional networks.

- 2. Understand why CNNs work the way they do – not just how to call Conv2D in a framework, but what the convolution and pooling operations are actually doing to an image.

- 3. Study the Vision Transformer architecture in enough depth to explain patch embedding, positional encoding, and multi-head self-attention without reach- ing for a slide deck.

- 4. Implement two independent baseline classifiers – DenseNet121 and ViT-B/16 – on a modern multi-class dermatology dataset (MILK10k), and compare them fairly under identical training conditions.

- 5. Reproduce a recent published multi-modal, explainable architecture (XAAI- Ledger) from its methodology section alone, including its Grad-CAM and SHAP explainability components.

- 6. Explore what a meaningful next step beyond this architecture could look like, grounded in the specific weaknesses observed during reproduction.

## 1.7 Internship Timeline

The seven weeks of the internship were structured to build up complexity gradu- ally – theory before implementation, single-modality baselines before multi-modal


fusion, and reproduction before proposing anything new. Table 1.1 and Figure 1.5 summarise this plan. [URL 🔗](#page-0)

*Table 1.1: Week-by-week internship work plan.*

| Week Work Done |
| --- |
| Week 1 Computer Vision Basics |
| Week 2 Vision Transformer |
| Week 3 DenseNet Baseline |
| Week 4 ViT Baseline |
| Week 5 XAAI-Ledger Implementation |
| Week 6 Proposed Improvements |
| Week 7 Analysis |

## Internship Timeline 7-Week Work Plan

*Figure 1.5: Seven-week internship timeline, from foundational computer vision concepts through to final analysis.*

The chapters that follow roughly mirror this timeline. Chapter 2 covers ev- erything learned in Weeks 1–2; the baseline implementations, the XAAI-Ledger reproduction, and the proposed improvements and analysis from the later weeks are covered in the chapters that follow it. [URL 🔗](#page-0)


## CHAPTER 2

## Background Study

This chapter is, essentially, seven weeks of notes cleaned up into something read- able. It has nothing to do with skin cancer specifically – that connection is made later. What it covers is the computer vision toolbox itself: how images were processed before deep learning existed, why convolutional networks became the default choice once deep learning arrived, and how Vision Transformers arrived to challenge that default. By the end of this chapter, every architecture used later in this report should feel like a natural extension of something explained here, rather than a black box pulled off a shelf.

## 2.1 Traditional Computer Vision

Before a neural network ever learns a single feature on its own, it is worth under- standing what a feature even is, because traditional computer vision made this completely explicit. An image, at its core, is just a grid of numbers – pixel inten- sities. Traditional CV techniques are, almost without exception, small mathemat- ical operations applied to this grid to pull out something structurally meaningful: edges, corners, blobs, textures.

## 2.1.1 Edge Detection

An edge is a place in the image where intensity changes sharply – the boundary of an object, say, or the border of a lesion. Mathematically, an edge corresponds to a large gradient in the image intensity function I(x, y). Three filters come up again and again:

Sobel operator. Approximates the image gradient in the x and y directions using two small 3 × 3 kernels:

The overall edge strength (gradient magnitude) at each pixel is then

|G| = q G2 x +G2 y.


Gaussian blur. Before edge detection, images are usually smoothed with a Gaussian kernel to suppress noise, since raw gradients are extremely sensitive to it:

Laplacian operator. Rather than a first-order gradient, the Laplacian is a second-order derivative operator, useful for detecting regions of rapid intensity change in all directions at once:

Figure 2.1 shows all three of these operators applied to the same test image, computed directly for this report rather than borrowed from a paper, so the reader can see exactly what each filter does to the same input. [URL 🔗](#page-0)

*Figure 2.1: Gaussian smoothing, Sobel edge magnitude, and Laplacian filtering applied to the same grayscale test image, computed for this report.*

The honest limitation of this entire family of methods is that a human had to decide, in advance, which filter matters. Sobel is good at edges. It is not good at detecting a subtle texture change that does not manifest as a sharp gradient. This is exactly the ceiling that deep learning eventually broke through – by learning the filters instead of hand-designing them.

## 2.2 Deep Learning Basics

An Artificial Neural Network (ANN) is, structurally, a sequence of layers, where each layer computes a weighted sum of its inputs followed by a non-linear activation function:

a(l) = σ

where W(l) and b(l) are the learnable weights and bias of layer l, and σ is a non- linearity such as ReLU. Stack enough of these layers and, in principle, the network can approximate extremely complex functions – this is the universal approximation

W(l)a(l−1) + b(l) ,


intuition behind deep learning.

The problem with applying a plain ANN directly to images is scale. A modestly sized 224 × 224 RGB image has 224 × 224 × 3 = 150,528 input values. A fully- connected first layer with even 1,000 neurons would need over 150 million weights just for that one layer – computationally wasteful, and worse, it throws away the fact that nearby pixels are related to each other. A Convolutional Neural Network (CNN) fixes both problems at once, by replacing full connectivity with small, shared filters that slide across the image (the same idea as the Sobel filter above, except now the filter’s numbers are learned rather than hand-picked). This gives CNNs two properties that make them well suited to images specifically:

-  Local connectivity – each neuron only looks at a small patch of the image, which matches the intuition that a pixel is most related to its neighbours.

-  Weight sharing – the same filter is reused across the entire image, drasti- cally reducing the number of parameters and letting the network detect the same pattern (e.g. an edge) regardless of where it appears.

## 2.3 Convolution Operation

The convolution operation itself is simple enough to compute by hand for a small example, which is usually the fastest way to actually understand it. For a single- channel input I and a kernel K of size k × k, the output at position (i, j) is

Three extra parameters control how this operation behaves in practice, and all three matter a great deal for the architectures used later in this report:

-  Kernel size – the spatial extent of the filter (commonly 3 × 3 or 5 × 5). Larger kernels see more context per step but cost more computation.

-  Stride – how far the kernel moves between applications. A stride of 2 roughly halves the output’s spatial dimensions.

-  Padding – whether the input is padded with zeros at the border, so the kernel can be centred on edge pixels too (“same” padding keeps the output size equal to the input size; “valid” padding shrinks it).

For an input of size W×W, kernel size F, stride S, and padding P, the output

spatial size is


*Figure 2.2 illustrates a 3 × 3 kernel sliding over a padded input with stride 1, producing one output feature map. [URL 🔗](#page-0)*

*Figure 2.2: A 3×3 kernel sliding with stride 1 over a zero-padded input, producing a single output feature map. The highlighted receptive field on the input maps to a single highlighted output pixel.*

Pooling. Convolutional layers are almost always followed by a pooling oper- ation, which reduces the spatial resolution of the feature map while keeping the most important information. Max pooling, the most common choice, simply takes the maximum value within each small window:

Pooling gives the network a small amount of translation invariance (a feature is still detected even if it shifts slightly) and reduces the parameter count of subsequent layers.

A typical CNN, then, is simply a repeated stack of Convolution → Activation → Pooling blocks, with the spatial size shrinking and the number of channels (feature types) growing at each stage, before a final classification head flattens everything into class probabilities.

## 2.4 Transfer Learning

Training a CNN or a ViT completely from scratch requires a very large labelled dataset – ImageNet, the standard benchmark, contains over a million labelled images across a thousand categories. Dermatology datasets, by comparison, are small. Transfer learning sidesteps this problem with a simple observation: the early layers of a network trained on natural images learn very general features (edges, colours, simple textures) that are useful for almost any visual task, not just the one the network was originally trained on.

The typical recipe is:


- 1. Pretraining – train a large CNN or ViT backbone on a large, general dataset (ImageNet).

- 2. Feature extraction – freeze the pretrained backbone’s weights and use it purely to convert an image into a feature vector.

- 3. Fine-tuning – optionally, unfreeze some or all of the backbone and continue training on the target dataset (here, dermoscopic images) at a low learning rate, so the general features adapt slightly to the new domain without being destroyed.

*Figure 2.3: The transfer-learning recipe used throughout this report: a frozen, ImageNet-pretrained backbone (CNN or ViT) acts as a general-purpose feature ex- tractor, feeding into a small trainable head adapted to the skin-cancer classification task.*

Every baseline model built during this internship – DenseNet121, ViT-B/16, and the EfficientNetB0 backbone used inside the XAAI-Ledger reproduction – follows exactly this pattern, and this is the reason it is possible to get reasonable performance on a dataset with only tens of thousands of images rather than a million.

## 2.5 Vision Transformer

The Vision Transformer (ViT) takes the transformer architecture, originally built for language, and applies it to images with a surprisingly small number of image- specific changes [4]. The core idea: treat an image as a sequence of patches, exactly the way a sentence is a sequence of words, and let self-attention handle the rest. [URL 🔗](#page-0)

## 2.5.1 Patch and Position Embedding

An input image of size H × W × C is split into N non-overlapping patches of size P × P, where N = HW P2 . Each flattened patch is linearly projected into a D-dimensional embedding:


where E is the learnable patch-embedding matrix, xclass is a special learnable CLS token prepended to the sequence (its final-layer representation is later used for classification), and Epos is a learnable positional embedding – necessary because, unlike a CNN, self-attention has no inherent notion of spatial order and needs to be told where each patch came from.

## 2.5.2 Transformer Encoder

The embedded sequence passes through L identical encoder blocks, each built from two sub-layers wrapped in residual connections and Layer Normalization:

where MSA is Multi-Head Self-Attention and MLP is a small feed-forward network (typically two linear layers with a GELU activation in between). The residual connections are what make it possible to stack many such blocks without gradients vanishing, and LayerNorm keeps the activations at each layer well-scaled during training.

## 2.5.3 Multi-Head Self-Attention

Self-attention is the mechanism that lets every patch look at every other patch. For a single attention head, each input token is projected into a Query, Key, and Value vector, and attention weights are computed as a scaled dot-product:

where dk is the dimensionality of the key vectors (the scaling factor prevents the dot products from growing too large and saturating the softmax). Multi-head attention simply runs h of these attention operations in parallel with different learned projections, and concatenates the results – allowing the model to attend to different types of relationships (e.g. colour similarity in one head, spatial proximity in another) simultaneously.

The upshot of all this machinery is that, from the very first encoder block, the CLS token can attend to every patch in the image directly. This is a fundamentally different inductive bias from a CNN, where a neuron in an early layer can only see a small local neighbourhood and has to wait for many layers of convolution before its receptive field covers the whole image.


*Figure 2.4: The Vision Transformer (ViT) pipeline: an image is split into patches, linearly embedded with a prepended CLS token and positional embeddings, passed through L transformer encoder blocks (each with multi-head self-attention, residual connections, and an MLP sub-layer), and the final CLS token representation is classified by a small MLP head.*

## 2.6 CNN vs ViT: A Theoretical Comparison

With both architectures now unpacked, it is worth putting them side by side before any experiment is run. Table 2.1 summarises the theoretical trade-offs; the empirical side of this comparison, on an actual dermatology dataset, is picked up again later in this report. [URL 🔗](#page-0)

Neither column in Table 2.1 is strictly better than the other – and that, ulti- mately, is the whole justification for the multi-modal, hybrid approach explored in the rest of this report. A CNN backbone is well suited to extracting fine local tex- ture from a dermoscopic image; a transformer-based fusion layer is well suited to relating that visual evidence to structured clinical metadata as a global reasoning step on top. Chapters that follow build directly on both halves of this table – first evaluating DenseNet121 (a pure CNN) against ViT-B/16 (a pure transformer) as independent baselines, and then examining an architecture that tries to combine the strengths of both. [URL 🔗](#page-0)


*Table 2.1: Theoretical comparison between Convolutional Neural Networks and Vision Transformers.*

| Aspect | CNN | ViT |
| --- | --- | --- |
| Inductive bias | Strong: locality, weight shar- | Weak: minimal built-in as- |
|   | ing, translation invariance | sumptions about image struc- |
|   |   | ture |
| Receptive field Local at early layers; grows |   | Global from the very first |
|   | gradually with depth | layer, via self-attention |
| Data require- | Performs well on small-to- | Typically needs large-scale |
| ment | medium datasets | pretraining data to match |
|   |   | CNN performance |
| Parameter effi- | Fewer parameters for a given | Generally more parameters; |
| ciency | receptive field, due to weight | self-attention is quadratic in |
|   | sharing | sequence length |
| Interpretability | Grad-CAM, saliency maps | Attention-map visualisation |
| tools | (well established) | (more direct, since attention |
|   |   | weights are inherently inter- |
|   |   | pretable) |
| Typical strength Fine, local texture and edge |   | Long-range dependencies and |
|   | patterns | global context |
| Computational | Convolution is linear in image | Self-attention scales quadrati- |
| cost | size | cally with number of patches |


```
References (partial – Chapters 1–2 only)
```


## Bibliography

- [1] J. M. Bae, H. H. Ku, H. Tokar, et al., “Diagnostic accuracy of dermoscopy for melanoma: a systematic review and meta-analysis,” British Journal of Dermatology, vol. 175, no. 6, pp. 1123–1131, 2016.

- [2] A. Romero-Lopez, X. Giro-i-Nieto, J. Burdick, and O. Marques, “Skin le- sion classification from dermoscopic images using deep learning techniques,” in Proc. IASTED International Conference on Biomedical Engineering, 2017.

- [3] M. Sharafudeen and S. S. Chandra, “Leveraging fusion of texture features and CASH-rule based architecture for interpretable dermatological diagnosis,” Biomedical Signal Processing and Control, 2023.

- [4] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Un- terthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words: Transformers for image recogni- tion at scale,” International Conference on Learning Representations (ICLR), 2021.
