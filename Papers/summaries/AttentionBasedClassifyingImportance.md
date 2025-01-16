# Attention-Based Methods

## Introduction
The introduction of the attention mechanism has been pivotal in the advancement of explainability in transformer-based models. Attention weights have emerged as one of the primary tools for understanding how models process inputs and make decisions. Two notable techniques, **Attention Rollout** and **Attention Flow**, provide frameworks for using attention weights to explain model behavior.

## Key Concepts

### Attention Weights
Attention weights indicate the importance of each token in the input relative to others when producing an output. They can be used as a **proxy for explanations** by analyzing their distribution across different layers and heads of the transformer.

- **Augmented Attention Weights**:
  To incorporate residual connections, attention weights are adjusted as follows:
  \[
  A^l = 0.5 \cdot W_{\text{att}}^l + 0.5 \cdot I
  \]
  where:
  - \( W_{\text{att}}^l \): The average of all attention weight matrices in the heads of layer \( l \).
  - \( I \): The identity matrix.

### Attention Rollout
This technique calculates cumulative attention matrices by chaining attention weights across layers:
\[
\tilde{A}^l =
\begin{cases}
A^l & \text{if it is the last layer}, \\
A^l \tilde{A}^{l-1} & \text{otherwise}.
\end{cases}
\]

- The final cumulative matrix can be visualized as a **heatmap**, showing the relevance of input tokens with respect to output tokens.

### Attention Flow
In this technique, the neural network is represented as a graph:
- Edges correspond to attention weights.
- Nodes represent input tokens (sources) and output tokens (sinks).
- Relevance is determined by the **max flow** of the graph, indicating how information flows between tokens.

## Applications

### Attention Weights as Relevance Scores
Several studies have employed attention weights as relevance scores across various domains:

1. **General Applications**:
   - Trisedya et al.: Explaining knowledge graph alignment.
   - Sebbaq and El Faddouli: Taxonomy-based classification of e-learning materials.
   - Chen et al.: Sequential recommendation models.

2. **Medical Applications**:
   - Graca et al.: Classifying Single Nucleotide Polymorphisms (SNPs).
   - Kim et al.: Explaining medical codes prediction.
   - Clauwaert et al.: Analyzing genomics transcription to assess head specialization.

3. **Other Use Cases**:
   - Wantiez et al.: Visual question answering in autonomous driving.
   - Schwenke et al.: Processing time series with symbolic abstraction.
   - Bacco et al.: Sentiment analysis using attention weights to select relevant input sentences.

### Attention Rollout

Attention Rollout has been used in various domains:
1. **Visual Object Tracking**:
   - Di Nardo et al.: Explainability in visual object-tracking tasks.

2. **Medical Diagnosis**:
   - Neto et al.: Detection of metaplasia using Attention Rollout combined with Grad-CAM.

3. **Process Monitoring**:
   - Pasquadibisceglie et al.: Generating heatmaps for next-activity prediction.

4. **COVID-19 Detection**:
   - Komorowski et al.: Comparing Attention Rollout with LIME and LRP for X-ray image analysis.

### Visualization of Attention Weights
Visualization of attention weights aids in explaining model outcomes and understanding their internal processes. Examples include:

1. **Tools and Methods**:
   - Fiok et al.: Used BertViz and TreeSHAP for transformer explainability.
   - Lal et al.: Proposed dimensionality reduction methods to visualize attention weights.

2. **Domain-Specific Applications**:
   - Neuroscience: Architecture for brain function analysis using fMRI data.
   - Genomics: Explaining DNA methylation site predictions.
   - Medicine: Eye disease classification from medical records and medical image segmentation.

3. **General Applications**:
   - Textual dialogue interaction.
   - Driver distraction identification.
   - Action recommendations.

### Visual Transformers
Transformers for image-related tasks have also leveraged attention weights for explainability:
- Ma et al.: Computed indiscriminative scores for image patches as a function of attention weights.

## Conclusion
Attention-based methods provide a versatile framework for interpreting and explaining transformer-based models. Techniques like Attention Rollout and Attention Flow, along with visualization tools, have been applied across diverse fields such as medicine, genomics, autonomous driving, and more, making them indispensable tools for modern machine learning interpretability.

