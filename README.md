# 📌 Multimodal Anomaly Detection with Vision-Language Models

**Author:** Abhiprabav Eppalapalli  
**Course:** DA623 – Winter 2025

---

## 💡 Motivation

Anomaly detection is essential in fields like manufacturing, healthcare, and cybersecurity. However, traditional systems usually rely on a single modality—such as image-based inspection or log analysis—which can miss critical signals when data is noisy or incomplete.

This project explores the use of **vision-language models**, particularly CLIP, to detect anomalies by comparing the **semantic similarity between images and their corresponding textual descriptions**. If the visual and textual inputs don’t align, the sample may be considered anomalous. This approach offers an intuitive and generalizable way to flag mismatches in multimodal data.

---

## 🧠 Background: Multimodal Learning and CLIP

Multimodal learning combines different types of inputs (e.g., text + image) to create richer and more robust representations. Vision-language models like **CLIP (Contrastive Language–Image Pretraining)** and **BLIP** have revolutionized this space by learning a **shared embedding space** for image-text pairs.

CLIP was trained on 400M image-caption pairs to align visual and textual information. It's primarily used for **zero-shot classification, retrieval, and captioning**, but here, we creatively apply it to detect **semantic mismatches**, which signal anomalies.

---

## 🔍 Problem Formulation

Given:
- An image `I` (e.g., a product or medical scan)
- A textual description `T` (e.g., log entry, label, or caption)

Goal:
- Encode both `I` and `T` using a vision-language model like CLIP.
- Compute the **cosine similarity** between the two embeddings.
- Flag samples with **low similarity** as potential anomalies.

This method is unsupervised, flexible, and generalizes across domains.

---
## 🧩 Key Learnings

- **Semantic Similarity as a Proxy for Anomaly:** Vision-language models allow us to detect mismatches in meaning, which often correlates with errors or anomalies.
- **Zero-shot Capabilities:** CLIP generalizes surprisingly well, even without fine-tuning.
- **Prompt Sensitivity:** The model is highly sensitive to the quality and phrasing of text inputs.
- **Multimodal > Unimodal:** Combining image and text improves robustness and interpretability.

---

## 🤔 Reflections

**What Surprised Me:**
- CLIP’s ability to identify subtle inconsistencies that even traditional models would miss.
- How little data is needed to start experimenting with powerful models thanks to HuggingFace and pretrained weights.

**Scope for Improvement:**
- Improve the **realism of text descriptions** by mining from actual log data.
- Add a **temporal dimension** (e.g., sensor trends over time).
- Use interpretability tools like **Grad-CAM** to understand what parts of the image influence similarity.

---

## 📚 References

- 🔗 [CLIP Paper (OpenAI)](https://arxiv.org/abs/2103.00020)  
- 🔗 [BLIP Paper](https://arxiv.org/abs/2201.12086)  
- 📂 [MVTec Anomaly Detection Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)  
- 💻 [CLIP GitHub Repo](https://github.com/openai/CLIP)  
- 🧠 [BAIR Blog on Visual Haystacks](https://bair.berkeley.edu/blog/2024/07/20/visual-haystacks/)

---
## 🧪 Code Demo

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Load model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load image and define candidate text descriptions
image = Image.open("sample_image.jpg")
texts = ["a clean metal nut", "a damaged metal nut", "a broken piece"]

# Encode inputs
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

# Calculate similarity
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

# Print results
for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob.item():.3f}")




> _This project was created as part of the DA623 Winter 2025 course at [IIT GUWAHATI]._
