Below is a comprehensive overview of *gradient-based methods* for model explainability in deep learning. These methods rely on **backpropagated gradients** to highlight or visualize how input features affect a model’s prediction. In other words, these techniques answer the question: *Given an input \(x\) and a class \(c\), where in \(x\) do changes most strongly affect the score \(Y_c\)?*

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [1. Introduction](#1-introduction)
- [2. Saliency (Simonyan et al.)](#2-saliency-simonyan-et-al)
  - [Why It Works](#why-it-works)
  - [Saliency Map Visualization](#saliency-map-visualization)
- [3. Guided Backpropagation (Springenberg et al.)](#3-guided-backpropagation-springenberg-et-al)
- [4. Input (\\times) Gradient (Kindermans et al.)](#4-input-times-gradient-kindermans-et-al)
  - [Why Multiply by (x)?](#why-multiply-by-x)
- [5. Contrastive Gradient Norm (Yin \& Neubig)](#5-contrastive-gradient-norm-yin--neubig)
- [6. Grad-CAM (Selvaraju et al.) and Variants](#6-grad-cam-selvaraju-et-al-and-variants)
  - [6.1 Grad-CAM++](#61-grad-cam)
- [7. Use Cases and Comparative Studies](#7-use-cases-and-comparative-studies)
  - [Example of a Saliency Map Comparison](#example-of-a-saliency-map-comparison)
- [8. Conclusion](#8-conclusion)
- [References](#references)

---

<a name="introduction"></a>
## 1. Introduction

Many neural-network training algorithms rely on **backpropagated gradients**, where the gradient of the loss function is computed with respect to the inputs (features). The idea can be extended to explainability: by examining gradients with respect to the inputs (or internal activations), we gain insights into **which features** (pixels, words, etc.) most heavily influence a model’s output.

In the sections below, we survey popular **gradient-based explanation methods**—beginning with Saliency Maps ([Simonyan et al. 2013][100]) and moving on to more sophisticated approaches, such as **Guided Backpropagation** ([Springenberg et al. 2015][101]), **Input \(\times\) Gradient** ([Kindermans et al. 2017][102]), **Contrastive Gradient Norm** ([Yin & Neubig 2017][103]), and **Grad-CAM** ([Selvaraju et al. 2017][104]) plus its extensions like **Grad-CAM++** ([Chattopadhyay et al. 2018][105]).

---

<a name="saliency"></a>
## 2. Saliency (Simonyan et al.)

> **Key idea**: *Use the gradient of the class score \(Y_c\) with respect to the input \(x\).*  

**Saliency** (sometimes called a *gradient-based saliency map*) was introduced by [Simonyan et al. in [100]](#references). The authors computed the gradient:
\[
\nabla_{x} Y_c
\]
and interpreted its magnitude (or absolute value) as an indicator of each input dimension’s importance. 

### Why It Works  
- By **linearizing** \(Y_c\) around a given input \(x\) (via a first-order Taylor series), the gradient acts like the **coefficients** in a linear model.  
- A larger gradient magnitude at a specific input feature implies that small changes in that feature would cause a larger change in \(Y_c\).

### Saliency Map Visualization  
- Often shown as a **heatmap** over the original image/text.  
- Highly positive or negative gradient magnitudes typically appear in red, while near-zero gradients might appear in blue or be transparent.

---

<a name="guided-backpropagation"></a>
## 3. Guided Backpropagation (Springenberg et al.)

> **Key idea**: *Only backpropagate “positive” gradients through the layers of a CNN while zeroing out others.*  

**Guided Backpropagation** was introduced by [Springenberg et al. in [101]](#references). This method involves a **forward pass** to a specified layer, then a **modified backward pass** where:  

1. Only the gradients corresponding to positive activations in the forward pass are retained.  
2. Negative gradient contributions are set to zero (or sometimes, positive ones are set to zero, depending on the variant).  

This “guidance” helps to reduce noise in the saliency map and often yields more visually interpretable explanations compared to plain saliency.

---

<a name="inputxgradient"></a>
## 4. Input \(\times\) Gradient (Kindermans et al.)

> **Key idea**: *Weight the gradient by the input value itself.*  

**Input \(\times\) Gradient** was introduced by [Kindermans et al. in [102]](#references). This technique scales the raw gradient by the original input \(x\):
\[
\text{InputXGradient}(x) = x \odot \nabla_{x} Y_c.
\]

### Why Multiply by \(x\)?  
- Multiplying by \(x\) emphasizes **regions of the input** that are both (1) large in magnitude and (2) influential according to the gradient.  
- Helps address the issue of high-gradient regions in areas of near-zero input, which might be less semantically meaningful.

---

<a name="contrastive-gradient-norm"></a>
## 5. Contrastive Gradient Norm (Yin & Neubig)

> **Key idea**: *Compute the gradient for the difference of two output classes.*  

[Yin & Neubig in [103]](#references) introduced the **Contrastive Gradient Norm**, which is defined as:
\[
\nabla_{x} \bigl(Y_c - Y_{c'}\bigr),
\]
where \(Y_c\) and \(Y_{c'}\) are the outputs (logits or softmax scores) of two different classes \(c\) and \(c'\). The motivation is to highlight **which features differentiate one specific class from another**.

They also extended the Input \(\times\) Gradient approach to this contrastive scenario, calling it the **Contrastive Input \(\times\) Gradient** method.

---

<a name="grad-cam-and-variants"></a>
## 6. Grad-CAM (Selvaraju et al.) and Variants

> **Key idea**: *Use global average pooling of the gradients at the last convolutional layer to build a class-discriminative localization map.*  

**Grad-CAM** ([Selvaraju et al. in [104]](#references)) generalizes the CAM (Class Activation Mapping) technique to **any CNN architecture**, not just those with a global average pooling layer before the final output. The method:

1. **Backpropagates** the gradient from output \(Y_c\) to the **last convolutional layer**.  
2. **Averages** these gradients across the spatial dimensions (width \(\times\) height) for each filter \(f\). This yields a set of weights \(\alpha_f\). Formally,  
   \[
   \alpha_f = \frac{1}{F}\sum_{i,j} \frac{\partial Y_c}{\partial V_{i,j,f}},
   \]
   where \(V_{i,j,f}\) is the activation of filter \(f\) at spatial location \((i, j)\), and \(F\) is the total number of spatial positions in that feature map.  
3. **Weights** the activation maps by these averaged gradients and sums them:
   \[
   \sum_{f} \alpha_{f} \cdot V_{x,y,f}.
   \]
4. **Applies ReLU**:
   \[
   R_{x,y,c} = \mathrm{ReLU}\bigl(\sum_{f}\bigl(\alpha_f \cdot V_{x,y,f}\bigr)\bigr).
   \]
   ReLU ensures only positive contributions remain, highlighting the regions that **positively contribute** to the target class.

<a name="grad-cam-plus-plus"></a>
### 6.1 Grad-CAM++

**Grad-CAM++** ([Chattopadhyay et al. in [105]](#references)) refines Grad-CAM by introducing **more precise weighting coefficients** for each spatial location in the last convolutional layer. Key differences:

- Moves the ReLU operation inside the partial derivative calculation.  
- Uses *different* weights per spatial location, rather than globally averaging them.  

This provides a **finer localization** of important regions in the input image (or feature map).

---

<a name="use-cases-and-comparative-studies"></a>
## 7. Use Cases and Comparative Studies

1. **COVID-19 Detection and Classification**  
   - [Sobahi et al. [106]](#references) used Grad-CAM on cough sound recordings.  
   - [Thon et al. [107]](#references) used Grad-CAM for 3-class severity classification of COVID-19 from chest radiographs.  
   - [Vaid et al. [108]](#references) used Grad-CAM for ECG 2D representation classification.  

2. **Medical Image Segmentation**  
   - [Wang et al. [109]](#references) used Grad-CAM++ in a transformer-based architecture for medical 3D image segmentation.

3. **Comparisons with Other Methods**  
   - [Wollek et al. [110]](#references) compared TMME and Grad-CAM for pneumothorax classification.  
   - [Neto et al. [81]](#references) used Grad-CAM and Attention Rollout to explain a model for metaplasia detection.  
   - [Thakur et al. [111]](#references) compared LIME and Grad-CAM for plant disease identification.  
   - [Kadir et al. [112]](#references) compared Soundness Saliency and Grad-CAM for image classification.  
   - [Vareille et al. [113]](#references) employed SHAP, Grad-CAM, Integrated Gradients, and Occlusion for multivariate time series analysis.  
   - [Hroub et al. [57]](#references) compared Grad-CAM, Grad-CAM++, Eigen-Grad-CAM, and AblationCAM for pneumonia and COVID-19 prediction.

4. **Transformers and Gradient Visualization**  
   - [Cornia et al. [114]](#references) proposed explaining transformers for visual captioning using saliency, Guided Backpropagation, and Integrated Gradients.  
   - [Poulton et al. [115]](#references) used saliency, InputXGradient, Integrated Gradients, Occlusion, and GradientSHAP for automated short-answer grading.  
   - [Alammar [58]](#references) introduced Ecco, a tool providing gradient-based visualizations for transformers.  
   - [Wang et al. [95]](#references) used attention weights and gradient-based visualization to explain outputs in a novel segmentation model.

### Example of a Saliency Map Comparison

Researchers often **visualize** these methods as **heatmaps** over the input image or text tokens. For instance, a comparison among Saliency, Guided Backpropagation, Grad-CAM, etc., might show **which parts of an image** are most responsible for a certain classification (Figure 7 in the original text).

---

<a name="conclusion"></a>
## 8. Conclusion

**Gradient-based explainability** methods provide a direct, **first-order** view of how inputs affect a model’s predictions. From simple **saliency** to more advanced **Grad-CAM** variants, these techniques remain a popular choice for **CNN** and **transformer** explainability, largely due to their ease of implementation (a single backprop pass) and **faithfulness** to the underlying model computations.

However, keep in mind:
- **Saliency** may be noisy or sensitive to small input perturbations.  
- **Guided Backprop** and **Input \(\times\) Gradient** mitigate some of these issues but can still produce partial or biased explanations.  
- **Grad-CAM** and **Grad-CAM++** provide class-specific localization maps, making them particularly valuable in vision tasks.  

In combination with other **perturbation-based** or **attention-based** methods, gradient-based explanations can yield a more comprehensive **explanation toolkit** for understanding and debugging modern deep learning models.

---

## References

> Below are the reference placeholders as mentioned in the original text. Please note that these reference numbers (e.g., [100]) map to citations in the quoted text and may differ from an official bibliography.

- [100] Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). *Deep inside convolutional networks: Visualising image classification models and saliency maps.* arXiv preprint arXiv:1312.6034.  
- [101] Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2015). *Striving for simplicity: The all convolutional net.* arXiv preprint arXiv:1412.6806.  
- [102] Kindermans, P.-J., Hooker, S., Adebayo, J., Alber, M., Schütt, K. T., Dhamdhere, K., … & Müller, K. (2017). *The (un)reliability of saliency methods*.  
- [103] Yin, Y., & Neubig, G. (2017). *A syntactic neural model for general-purpose code generation.* In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics* (Vol. 1, pp. 440–450).  
- [104] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 618–626).  
- [105] Chattopadhyay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2018). *Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks.* In *2018 IEEE Winter Conference on Applications of Computer Vision (WACV)* (pp. 839–847). IEEE.  
- [106] Sobahi, N. et al. — *COVID-19 cough sound recordings.*  
- [107] Thon, S. et al. — *3-class severity classification of COVID-19.*  
- [108] Vaid, A. et al. — *ECG 2D representation classification.*  
- [109] Wang, et al. — *Transformer-based architecture for medical 3D image segmentation.*  
- [110] Wollek, L. et al. — *TMME vs. Grad-CAM for pneumothorax classification.*  
- [81]  Neto, E. C. et al. — *Grad-CAM & Attention Rollout for metaplasia detection.*  
- [111] Thakur, A. et al. — *LIME vs. Grad-CAM for plant disease identification.*  
- [112] Kadir, T. et al. — *Soundness Saliency & Grad-CAM for image classification.*  
- [113] Vareille, A. et al. — *SHAP, Grad-CAM, Integrated Gradients, Occlusion for time series.*  
- [57]  Hroub, A. M. et al. — *Grad-CAM vs. Grad-CAM++ vs. Eigen-Grad-CAM vs. AblationCAM.*  
- [114] Cornia, M., Stefanini, M., Baraldi, L., & Cucchiara, R. (2020). *M2: Meshed-Memory Transformer for Image Captioning.* In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*.  
- [115] Poulton, T., Shum, S., & others — *Transformers for automated short-answer grading.*  
- [58]  Alammar, J. — *Ecco library for transformer visualization.*  
- [95]  Wang, et al. — *Gradient-based visualization in a novel segmentation model.*  

---

**Citation Note:**  
If you use these methods, please cite the original authors. The reference numbers in this document correspond to the text excerpt provided and may not match an official bibliography exactly.